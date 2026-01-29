import json
import numpy as np
from collections import defaultdict
import faiss
import heapq
from utils import get_timestamp, generate_id, get_embedding, normalize_vector, llm_extract_keywords
from datetime import datetime
from utils import OpenAIClient
from utils import get_timestamp, generate_id, get_embedding, normalize_vector, llm_extract_keywords, compute_time_decay

# 初始化 OpenAI 客户端，用于 eval 目录下的特定逻辑
client = OpenAIClient(
    api_key='',
    base_url='https://cn2us02.opapi.win/v1'
)

# 计算记忆条目的新鲜度权重，采用指数衰减算法
def compute_recency(last_visit_time, tau=24):
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    now = datetime.now()
    t1 = datetime.strptime(last_visit_time, fmt)
    delta_hours = (now - t1).total_seconds() / 3600.0
    # 衰减周期 tau 控制衰减速度
    return np.exp(- delta_hours / tau)

# 计算记忆段（Session）的热度评分
def compute_segment_heat(session, alpha=0.8, beta=0.8, gamma=0.0001):
    # 综合考虑访问频次、交互规模和最后访问的新鲜度
    N_visit = session.get("N_visit", 0)
    L_interaction = session.get("L_interaction", 0)
    R_recency = session.get("R_recency", 1.0)
    return alpha * N_visit + beta * L_interaction + gamma * R_recency

# 中期记忆管理类，负责会话分段存储与热度淘汰
class MidTermMemory:
    # 初始化中期记忆，配置最大容量和数据文件路径
    def __init__(self, max_capacity=7, file_path="mid_term.json"):
        self.max_capacity = max_capacity
        self.file_path = file_path
        self.sessions = {}
        # 记录每个会话的访问频率，用于 LFU 淘汰
        self.access_frequency = defaultdict(int)
        # 维护一个热度堆，用于高效查找最热记忆
        self.heap = []
        self.load()

    # 通过页面 ID 跨会话查找具体的对话页面
    def get_page_by_id(self, page_id):
        for session in self.sessions.values():
            for page in session["details"]:
                if page["page_id"] == page_id:
                    return page
        return None

    # 建立或更新两个对话页面之间的链式关联关系
    def update_page_connections(self, prev_page_id, next_page_id):
        if prev_page_id:
            prev_page = self.get_page_by_id(prev_page_id)
            if prev_page:
                prev_page["next_page"] = next_page_id
        if next_page_id:
            next_page = self.get_page_by_id(next_page_id)
            if next_page:
                next_page["pre_page"] = prev_page_id

    # 执行 LFU 淘汰策略，删除访问频率最低的会话
    def evict_lfu(self):
        if not self.access_frequency:
            return
        
        # 寻找访问次数最少的会话 ID
        lfu_sid = min(self.access_frequency, key=self.access_frequency.get)
        print(f"中期记忆：LFU 淘汰会话段 {lfu_sid}。")
        
        if lfu_sid not in self.sessions:
            del self.access_frequency[lfu_sid]
            return
        
        session_to_delete = self.sessions[lfu_sid]
        # 在删除前，确保被断开的页面逻辑链条能重新缝合
        for page in session_to_delete["details"]:
            prev_page_id = page.get("pre_page")
            next_page_id = page.get("next_page")
            self.update_page_connections(prev_page_id, next_page_id)
        
        del self.sessions[lfu_sid]
        del self.access_frequency[lfu_sid]
        self.save()
        self.rebuild_heap()

    # 向中期记忆添加一个全新的对话会话段
    def add_session(self, summary, details):
        session_id = generate_id("session")
        # 生成会话摘要的向量及关键词
        summary_vec = get_embedding(summary)
        summary_vec = normalize_vector(summary_vec).tolist()
        summary_keywords = list(llm_extract_keywords(summary, client=client))
        
        new_details = []
        # 逐页处理对话详情
        for page in details:
            if "page_id" not in page:
                page["page_id"] = generate_id("page")
            full_text = f"User: {page.get('user_input','')} Assiant: {page.get('agent_response','')}"
            # 为每一页对话生成向量和关键词
            inp_vec = get_embedding(full_text)
            inp_vec = normalize_vector(inp_vec).tolist()
            page_keywords = list(llm_extract_keywords(full_text, client=client))
            page["page_embedding"] = inp_vec
            page["page_keywords"] = page_keywords
            page["preloaded"] = False
            page["analyzed"] = False
            # page["pre_page"] = None
            # page["next_page"] = None
            # page["meta_info"] = None
            new_details.append(page)
        
        # 构造会话对象，包含统计信息和热度指标
        session_obj = {
            "id": session_id,
            "summary": summary,
            "summary_keywords": summary_keywords,
            "summary_embedding": summary_vec,
            "details": new_details,
            "L_interaction": len(new_details),
            "R_recency": 1.0,
            "N_visit": 0,
            "H_segment": 0.0,
            "timestamp": get_timestamp(),
            "access_count": 0
        }
        self.sessions[session_id] = session_obj
        # 计算并更新热度，推入优先级队列
        session_obj["H_segment"] = compute_segment_heat(session_obj)
        self.access_frequency[session_id] = 0
        heapq.heappush(self.heap, (-session_obj["H_segment"], session_id))
        print(f"中期记忆：新增会话段 {session_id}，初始热度 {session_obj['H_segment']:.2f}。")
        # 容量超出时触发 LFU 淘汰
        if len(self.sessions) > self.max_capacity:
            self.evict_lfu()
        self.save()
        return session_id

    # 重构热度堆，确保堆序正确
    def rebuild_heap(self):
        self.heap = [(-session["H_segment"], sid) for sid, session in self.sessions.items()]
        heapq.heapify(self.heap)

    # 根据语义相似度，将新页面插入最匹配的现有会话中
    def insert_pages_into_session(self, summary, keyworks, pages, similarity_threshold=0.6, alpha=1.0):
        new_summary_vec = get_embedding(summary)
        new_summary_vec = normalize_vector(new_summary_vec)
        new_keywords = keyworks
        
        best_sid = None
        best_sim = -1
        # 语义检索：寻找最匹配的会话段
        for sid, session in self.sessions.items():
            sv = np.array(session["summary_embedding"], dtype=np.float32)
            sim = float(np.dot(sv, new_summary_vec))
            if sim > best_sim:
                best_sim = sim
                best_sid = sid
        
        # 如果找到候选，则计算综合评分（语义 + 关键词重合度）
        if best_sim >= 0 and best_sid is not None:
            print(f"中期记忆：尝试合并到会话段 {best_sid}（摘要相似度 {best_sim:.2f}）。")
            session = self.sessions[best_sid]
            session_keywords = set(session.get("summary_keywords", []))
            new_kw_set = set(new_keywords)
            if session_keywords and new_kw_set:
                overlap = session_keywords & new_kw_set
                s_top = 0.5 * (len(overlap)/len(session_keywords) + len(overlap)/len(new_kw_set))
            else:
                s_top = 0
            overall_score = best_sim + alpha * s_top
            
            # 达到阈值则合并页面
            if overall_score >= similarity_threshold:
                print(f"中期记忆：综合得分 {overall_score:.2f} 满足合并条件，将页面追加。")
                for p in pages:
                    if "page_id" not in p:
                        p["page_id"] = generate_id("page")
                    full_text = f"用户: {p.get('user_input','')}"
                    vec = get_embedding(full_text)
                    vec = normalize_vector(vec).tolist()
                    p["page_embedding"] = vec
                    p["page_keywords"] = keyworks
                    p["preloaded"] = False
                    # p["pre_page"] = None
                    # p["next_page"] = None
                    # p["meta_info"] = None
                    session["details"].append(p)
                session["timestamp"] = get_timestamp()
            else:
                # 不满足阈值，则视为独立主题，新增会话段
                print("中期记忆：综合得分不足，新增会话段。")
                self.add_session(summary, pages)
        else:
            # 没有历史会话，直接新建
            print("中期记忆：无相似会话段，新建会话段。")
            self.add_session(summary, pages)
        
        # 更新对应会话段的交互长度和最新热度
        if best_sid is not None and best_sid in self.sessions:
            session = self.sessions[best_sid]
            session["L_interaction"] += len(pages)
            session["H_segment"] = compute_segment_heat(session)
        
        self.rebuild_heap()
        self.save()

    # 通过会话摘要搜索并定位最相关的记忆片段及其页面
    def search_sessions_by_summary(self, query, client, segment_threshold=0.8, page_threshold=0.7, top_k=5, tau=3600, gamma=0.5, alpha=1.0):
        if not self.sessions:
            return []
        
        # 使用 FAISS 构建摘要向量索引进行初筛
        session_ids = list(self.sessions.keys())
        embeddings = np.array([self.sessions[s]["summary_embedding"] for s in session_ids], dtype=np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        query_vec = get_embedding(query)
        query_vec = normalize_vector(query_vec)
        query_arr = np.array([query_vec], dtype=np.float32)
        # 返回相似度排名前 top_k 的会话
        distances, indices = index.search(query_arr, top_k)
        
        query_keywords = llm_extract_keywords(query, client)
        current_time = datetime.now()
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            sid = session_ids[idx]
            session = self.sessions[sid]
            session_time = datetime.strptime(session["timestamp"], "%Y-%m-%d %H:%M:%S")
            delta = (current_time - session_time).total_seconds()
            #lambda_t = np.exp(-delta/tau)
            lambda_t=1
            
            # 细化计算摘要相似度（向量余弦相似度 + 关键词加权）
            session_keywords = set(session.get("summary_keywords", []))
            if query_keywords and session_keywords:
                overlap = query_keywords & session_keywords
                s_top = 0.5 * (len(overlap)/len(query_keywords) + len(overlap)/len(session_keywords))
            else:
                s_top = 0
            
            overall = lambda_t * (dist + alpha * s_top)
            
            # 如果会话层级匹配成功，则进一步在会话内部搜索页面
            if overall >= segment_threshold:
                matched_pages = []
                for page in session["details"]:
                    full_text = f"{page.get('user_input','')}{page.get('timestamp','')}{page.get('agent_response','')}"
                    pvec = np.array(get_embedding(full_text), dtype=np.float32)
                    pvec = normalize_vector(pvec)
                    sim_page = float(np.dot(pvec, query_vec))
                    # 筛选出页面层级也满足阈值的对话
                    if sim_page >= page_threshold:
                        matched_pages.append([page, sim_page])
                
                # 如果会话内部有页面命中，则更新该会话的热度指标
                if matched_pages:
                    self.access_frequency[sid] += 1
                    session["N_visit"] += 1
                    session["last_visit_time"] = get_timestamp()
                    session["R_recency"] = compute_time_decay(session["last_visit_time"], get_timestamp(), tau)
                    session["access_count"] += 1
                    session["H_segment"] = compute_segment_heat(session)
                    print(f"中期记忆：会话段 {sid} 命中，匹配 {len(matched_pages)} 个 QA 对。")
                    results.append({
                        "session": session,
                        "matched_pages": matched_pages,
                        "session_similarity": overall
                    })
        
        return results

    # 持久化中期记忆数据及其访问统计
    def save(self):
        sessions_to_save = {}
        for sid, session in self.sessions.items():
            sessions_to_save[sid] = session
        with open(self.file_path, "w", encoding="utf-8") as f:
            data = {"sessions": sessions_to_save, "access_frequency": dict(self.access_frequency)}
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("中期记忆：保存成功。")

    # 从磁盘加载中期记忆
    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.sessions = data.get("sessions", {})
                self.access_frequency = defaultdict(int, data.get("access_frequency", {}))
            self.rebuild_heap()
            print("中期记忆：加载成功。")
        except Exception:
            self.sessions = {}
            self.access_frequency = defaultdict(int)
            self.heap = []
            print("中期记忆：无历史数据。")
