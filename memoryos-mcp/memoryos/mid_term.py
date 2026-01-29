import json
import numpy as np
from collections import defaultdict
import faiss
import heapq
from datetime import datetime

try:
    # 尝试相对导入核心工具函数
    from .utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector, 
        compute_time_decay, ensure_directory_exists, OpenAIClient
    )
except ImportError:
    # 回退到绝对导入
    from utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector, 
        compute_time_decay, ensure_directory_exists, OpenAIClient
    )

# 热度计算常量（可调整或配置）
# Heat computation constants (can be tuned or made configurable)
HEAT_ALPHA = 1.0
HEAT_BETA = 1.0
HEAT_GAMMA = 1
# 计算热度时的衰减周期（小时）
RECENCY_TAU_HOURS = 24 # For R_recency calculation in compute_segment_heat

# 计算会话片段的热度，综合考虑访问次数、交互长度和时间衰减因素
def compute_segment_heat(session, alpha=HEAT_ALPHA, beta=HEAT_BETA, gamma=HEAT_GAMMA, tau_hours=RECENCY_TAU_HOURS):
    N_visit = session.get("N_visit", 0)
    L_interaction = session.get("L_interaction", 0)
    
    # 根据最后访问时间计算新鲜度
    # Calculate recency based on last_visit_time
    R_recency = 1.0 # Default if no last_visit_time
    if session.get("last_visit_time"):
        R_recency = compute_time_decay(session["last_visit_time"], get_timestamp(), tau_hours)
    
    # 更新会话的新鲜度因子
    session["R_recency"] = R_recency # Update session's recency factor
    return alpha * N_visit + beta * L_interaction + gamma * R_recency

# 中期记忆类，管理从短期记忆整合而来的会话片段
class MidTermMemory:
    # 初始化中期记忆，设置存储路径、容量和嵌入模型
    def __init__(self, file_path: str, client: OpenAIClient, max_capacity=2000, embedding_model_name: str = "all-MiniLM-L6-v2", embedding_model_kwargs: dict = None):
        self.file_path = file_path
        # 确保存储路径存在
        ensure_directory_exists(self.file_path)
        self.client = client
        self.max_capacity = max_capacity
        self.sessions = {} # {session_id: session_object}
        # 用于 LFU（最近最少使用）淘汰策略的访问频率统计
        self.access_frequency = defaultdict(int) # {session_id: access_count_for_lfu}
        # 最小堆，存储 (-热度, 会话ID)，用于快速找到最热的片段
        self.heap = []  # Min-heap storing (-H_segment, session_id) for hottest segments

        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs if embedding_model_kwargs is not None else {}
        # 加载磁盘上的中期记忆数据
        self.load()

    # 根据页面 ID 获取特定页面数据
    def get_page_by_id(self, page_id):
        for session in self.sessions.values():
            for page in session.get("details", []):
                if page.get("page_id") == page_id:
                    return page
        return None

    # 更新页面之间的前后链接关系
    def update_page_connections(self, prev_page_id, next_page_id):
        if prev_page_id:
            prev_page = self.get_page_by_id(prev_page_id)
            if prev_page:
                prev_page["next_page"] = next_page_id
        if next_page_id:
            next_page = self.get_page_by_id(next_page_id)
            if next_page:
                next_page["pre_page"] = prev_page_id
        # self.save() # Avoid saving on every minor update; save at higher level operations

    # 执行 LFU 淘汰，删除访问频率最低的会话以释放空间
    def evict_lfu(self):
        if not self.access_frequency or not self.sessions:
            return
        
        # 找到访问频率最小的会话 ID
        lfu_sid = min(self.access_frequency, key=self.access_frequency.get)
        print(f"MidTermMemory: LFU eviction. Session {lfu_sid} has lowest access frequency.")
        
        if lfu_sid not in self.sessions:
            # 如果会话已经不在，清理访问频率记录并重构堆
            del self.access_frequency[lfu_sid] # Clean up access frequency if session already gone
            self.rebuild_heap()
            return
        
        # 从会话字典中移除
        session_to_delete = self.sessions.pop(lfu_sid) # Remove from sessions
        del self.access_frequency[lfu_sid] # Remove from LFU tracking

        # 清理页面链接关系
        # Clean up page connections if this session's pages were linked
        for page in session_to_delete.get("details", []):
            prev_page_id = page.get("pre_page")
            next_page_id = page.get("next_page")
            # If a page from this session was linked to an external page, nullify the external link
            if prev_page_id and not self.get_page_by_id(prev_page_id): # Check if prev page is still in memory
                 # This case should ideally not happen if connections are within sessions or handled carefully
                 pass 
            if next_page_id and not self.get_page_by_id(next_page_id):
                 pass
            # More robustly, one might need to search all other sessions if inter-session linking was allowed
            # For now, assuming internal consistency or that MemoryOS class manages higher-level links

        # 重构热度堆并保存状态
        self.rebuild_heap()
        self.save()
        print(f"MidTermMemory: Evicted session {lfu_sid}.")

    # 添加新的会话，包括摘要、详细页面以及计算嵌入向量和热度
    def add_session(self, summary, details, summary_keywords=None):
        session_id = generate_id("session")
        # 计算摘要的嵌入向量
        summary_vec = get_embedding(
            summary, 
            model_name=self.embedding_model_name, 
            **self.embedding_model_kwargs
        )
        # 归一化向量并转换为列表存储
        summary_vec = normalize_vector(summary_vec).tolist()
        summary_keywords = summary_keywords if summary_keywords is not None else []
        
        processed_details = []
        for page_data in details:
            page_id = page_data.get("page_id", generate_id("page"))
            
            # 检查是否已有嵌入向量，避免重复计算
            # 检查是否已有embedding，避免重复计算
            if "page_embedding" in page_data and page_data["page_embedding"]:
                print(f"MidTermMemory: Reusing existing embedding for page {page_id}")
                inp_vec = page_data["page_embedding"]
                # 确保embedding是normalized的
                if isinstance(inp_vec, list):
                    inp_vec_np = np.array(inp_vec, dtype=np.float32)
                    # 检查是否需要重新归一化
                    if np.linalg.norm(inp_vec_np) > 1.1 or np.linalg.norm(inp_vec_np) < 0.9:  # 检查是否需要重新normalize
                        inp_vec = normalize_vector(inp_vec_np).tolist()
            else:
                # 计算新页面的嵌入向量
                print(f"MidTermMemory: Computing new embedding for page {page_id}")
                full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                inp_vec = get_embedding(
                    full_text,
                    model_name=self.embedding_model_name,
                    **self.embedding_model_kwargs
                )
                inp_vec = normalize_vector(inp_vec).tolist()
            
            # 处理页面关键词
            # 使用已有keywords或设置为空（由multi-summary提供）
            if "page_keywords" in page_data and page_data["page_keywords"]:
                print(f"MidTermMemory: Using existing keywords for page {page_id}")
                page_keywords = page_data["page_keywords"]
            else:
                print(f"MidTermMemory: Setting empty keywords for page {page_id} (will be filled by multi-summary)")
                page_keywords = []
            
            processed_page = {
                **page_data, # Carry over existing fields like user_input, agent_response, timestamp
                "page_id": page_id,
                "page_embedding": inp_vec,
                "page_keywords": page_keywords,
                "preloaded": page_data.get("preloaded", False), # Preserve if passed
                "analyzed": page_data.get("analyzed", False),   # Preserve if passed
                # pre_page, next_page, meta_info are handled by DynamicUpdater
            }
            processed_details.append(processed_page)
        
        current_ts = get_timestamp()
        # 构建会话对象
        session_obj = {
            "id": session_id,
            "summary": summary,
            "summary_keywords": summary_keywords,
            "summary_embedding": summary_vec,
            "details": processed_details,
            "L_interaction": len(processed_details),
            "R_recency": 1.0, # Initial recency
            "N_visit": 0,
            "H_segment": 0.0, # Initial heat, will be computed
            "timestamp": current_ts, # Creation timestamp
            "last_visit_time": current_ts, # Also initial last_visit_time for recency calc
            "access_count_lfu": 0 # For LFU eviction policy
        }
        # 计算初始热度并推入堆中
        session_obj["H_segment"] = compute_segment_heat(session_obj)
        self.sessions[session_id] = session_obj
        self.access_frequency[session_id] = 0 # Initialize for LFU
        # 使用负热度实现大顶堆行为
        heapq.heappush(self.heap, (-session_obj["H_segment"], session_id)) # Use negative heat for max-heap behavior
        
        print(f"MidTermMemory: Added new session {session_id}. Initial heat: {session_obj['H_segment']:.2f}.")
        # 如果超过最大容量，则触发淘汰
        if len(self.sessions) > self.max_capacity:
            self.evict_lfu()
        # 保存中期记忆到磁盘
        self.save()
        return session_id

    # 从当前会话数据重构热度堆
    def rebuild_heap(self):
        self.heap = []
        for sid, session_data in self.sessions.items():
            # Ensure H_segment is up-to-date before rebuilding heap if necessary
            # session_data["H_segment"] = compute_segment_heat(session_data)
            heapq.heappush(self.heap, (-session_data["H_segment"], sid))
        # heapq.heapify(self.heap) # Not needed if pushing one by one
        # No save here, it's an internal operation often followed by other ops that save

    # 将新页面插入现有会话或创建新会话（基于摘要相似度）
    def insert_pages_into_session(self, summary_for_new_pages, keywords_for_new_pages, pages_to_insert, 
                                  similarity_threshold=0.6, keyword_similarity_alpha=1.0):
        # 如果没有现有会话，则直接创建
        if not self.sessions: # If no existing sessions, just add as a new one
            print("MidTermMemory: No existing sessions. Adding new session directly.")
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

        # 计算新摘要的嵌入向量
        new_summary_vec = get_embedding(
            summary_for_new_pages,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs
        )
        new_summary_vec = normalize_vector(new_summary_vec)
        
        best_sid = None
        best_overall_score = -1

        # 遍历现有会话，寻找最匹配的会话进行合并
        for sid, existing_session in self.sessions.items():
            existing_summary_vec = np.array(existing_session["summary_embedding"], dtype=np.float32)
            # 计算语义相似度（点积）
            semantic_sim = float(np.dot(existing_summary_vec, new_summary_vec))
            
            # 基于杰卡德系数计算关键词相似度
            # Keyword similarity (Jaccard index based)
            existing_keywords = set(existing_session.get("summary_keywords", []))
            new_keywords_set = set(keywords_for_new_pages)
            s_topic_keywords = 0
            if existing_keywords and new_keywords_set:
                intersection = len(existing_keywords.intersection(new_keywords_set))
                union = len(existing_keywords.union(new_keywords_set))
                if union > 0:
                    s_topic_keywords = intersection / union 
            
            # 综合评分
            overall_score = semantic_sim + keyword_similarity_alpha * s_topic_keywords
            
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_sid = sid
        
        # 如果最高分超过阈值，则合并到对应会话
        if best_sid and best_overall_score >= similarity_threshold:
            print(f"MidTermMemory: Merging pages into session {best_sid}. Score: {best_overall_score:.2f} (Threshold: {similarity_threshold})")
            target_session = self.sessions[best_sid]
            
            processed_new_pages = []
            for page_data in pages_to_insert:
                page_id = page_data.get("page_id", generate_id("page")) # Use existing or generate new ID
                
                # 检查是否已有嵌入向量，避免重复计算
                # 检查是否已有embedding，避免重复计算
                if "page_embedding" in page_data and page_data["page_embedding"]:
                    print(f"MidTermMemory: Reusing existing embedding for page {page_id}")
                    inp_vec = page_data["page_embedding"]
                    # 确保embedding是normalized的
                    if isinstance(inp_vec, list):
                        inp_vec_np = np.array(inp_vec, dtype=np.float32)
                        if np.linalg.norm(inp_vec_np) > 1.1 or np.linalg.norm(inp_vec_np) < 0.9:  # 检查是否需要重新normalize
                            inp_vec = normalize_vector(inp_vec_np).tolist()
                else:
                    # 为新页面计算嵌入向量
                    print(f"MidTermMemory: Computing new embedding for page {page_id}")
                    full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                    inp_vec = get_embedding(
                        full_text,
                        model_name=self.embedding_model_name,
                        **self.embedding_model_kwargs
                    )
                    inp_vec = normalize_vector(inp_vec).tolist()
                
                # 处理页面关键词，继承或使用现有关键词
                # 使用已有keywords或继承session的keywords
                if "page_keywords" in page_data and page_data["page_keywords"]:
                    print(f"MidTermMemory: Using existing keywords for page {page_id}")
                    page_keywords_current = page_data["page_keywords"]
                else:
                    print(f"MidTermMemory: Using session keywords for page {page_id}")
                    page_keywords_current = keywords_for_new_pages

                processed_page = {
                    **page_data, # Carry over existing fields
                    "page_id": page_id,
                    "page_embedding": inp_vec,
                    "page_keywords": page_keywords_current,
                    # analyzed, preloaded flags should be part of page_data if set
                }
                target_session["details"].append(processed_page)
                processed_new_pages.append(processed_page)

            # 更新会话统计信息和热度
            target_session["L_interaction"] += len(pages_to_insert)
            target_session["last_visit_time"] = get_timestamp() # Update last visit time on modification
            target_session["H_segment"] = compute_segment_heat(target_session)
            # 重构热度堆并保存
            self.rebuild_heap() # Rebuild heap as heat has changed
            self.save()
            return best_sid
        else:
            # 未找到合适会话，创建新会话
            print(f"MidTermMemory: No suitable session to merge (best score {best_overall_score:.2f} < threshold {similarity_threshold}). Creating new session.")
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

    # 搜索中期记忆中的会话和页面
    def search_sessions(self, query_text, segment_similarity_threshold=0.0, page_similarity_threshold=0.0,
                          top_k_sessions=6, keyword_alpha=1.0, recency_tau_search=3600):
        if not self.sessions:
            return []

        # 计算查询文本的嵌入向量
        query_vec = get_embedding(
            query_text,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec)
        # 当前暂未提取关键词，仅依赖语义相似度
        query_keywords = set()  # Keywords extraction removed, relying on semantic similarity

        candidate_sessions = []
        session_ids = list(self.sessions.keys())
        if not session_ids: return []

        # 获取所有会话摘要的嵌入向量并使用 FAISS 进行相似度搜索
        summary_embeddings_list = [self.sessions[s]["summary_embedding"] for s in session_ids]
        summary_embeddings_np = np.array(summary_embeddings_list, dtype=np.float32)

        dim = summary_embeddings_np.shape[1]
        # 使用内积进行相似度计算
        index = faiss.IndexFlatIP(dim) # Inner product for similarity
        index.add(summary_embeddings_np)
        
        query_arr_np = np.array([query_vec], dtype=np.float32)
        # 搜索最相关的 top_k 个会话
        distances, indices = index.search(query_arr_np, min(top_k_sessions, len(session_ids)))

        results = []
        current_time_str = get_timestamp()

        # 遍历检索结果
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            session_id = session_ids[idx]
            session = self.sessions[session_id]
            semantic_sim_score = float(distances[0][i]) # This is the dot product

            # 计算会话摘要的关键词相似度
            # Keyword similarity for session summary
            session_keywords = set(session.get("summary_keywords", []))
            s_topic_keywords = 0
            if query_keywords and session_keywords:
                intersection = len(query_keywords.intersection(session_keywords))
                union = len(query_keywords.union(session_keywords))
                if union > 0: s_topic_keywords = intersection / union

            # 计算会话相关性总分
            # Time decay for session recency in search scoring
            # time_decay_factor = compute_time_decay(session["timestamp"], current_time_str, tau_hours=recency_tau_search)
            
            # Combined score for session relevance
            session_relevance_score =  (semantic_sim_score + keyword_alpha * s_topic_keywords)

            # 如果会话相关性得分超过阈值，则进一步搜索会话内的页面
            if session_relevance_score >= segment_similarity_threshold:
                matched_pages_in_session = []
                for page in session.get("details", []):
                    page_embedding = np.array(page["page_embedding"], dtype=np.float32)
                    # page_keywords = set(page.get("page_keywords", []))

                    # 计算页面相似度得分
                    page_sim_score = float(np.dot(page_embedding, query_vec))
                    # Can also add keyword sim for pages if needed, but keeping it simpler for now

                    if page_sim_score >= page_similarity_threshold:
                        matched_pages_in_session.append({"page_data": page, "score": page_sim_score})
                
                # 如果会话内有匹配页面，则更新会话访问统计并重构堆
                if matched_pages_in_session:
                    # Update session access stats
                    session["N_visit"] += 1
                    session["last_visit_time"] = current_time_str
                    session["access_count_lfu"] = session.get("access_count_lfu", 0) + 1
                    self.access_frequency[session_id] = session["access_count_lfu"]
                    session["H_segment"] = compute_segment_heat(session)
                    self.rebuild_heap() # Heat changed
                    
                    results.append({
                        "session_id": session_id,
                        "session_summary": session["summary"],
                        "session_relevance_score": session_relevance_score,
                        "matched_pages": sorted(matched_pages_in_session, key=lambda x: x["score"], reverse=True) # Sort pages by score
                    })
        
        # 保存访问更新后的状态
        self.save() # Save changes from access updates
        # Sort final results by session_relevance_score
        return sorted(results, key=lambda x: x["session_relevance_score"], reverse=True)

    # 将当前中期记忆状态保存到磁盘 JSON 文件
    def save(self):
        # 构建待保存的数据字典
        # Make a copy for saving to avoid modifying heap during iteration if it happens
        # Though current heap is list of tuples, so direct modification risk is low
        # sessions_to_save = {sid: data for sid, data in self.sessions.items()}
        data_to_save = {
            "sessions": self.sessions,
            "access_frequency": dict(self.access_frequency), # Convert defaultdict to dict for JSON
            # Heap is derived, no need to save typically, but can if desired for faster load
            # "heap_snapshot": self.heap 
        }
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving MidTermMemory to {self.file_path}: {e}")

    # 从磁盘 JSON 文件加载中期记忆
    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.sessions = data.get("sessions", {})
                self.access_frequency = defaultdict(int, data.get("access_frequency", {}))
                # 加载后根据会话热度重构堆
                self.rebuild_heap() # Rebuild heap from loaded sessions
            print(f"MidTermMemory: Loaded from {self.file_path}. Sessions: {len(self.sessions)}.")
        except FileNotFoundError:
            print(f"MidTermMemory: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            print(f"MidTermMemory: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            print(f"MidTermMemory: An unexpected error occurred during load from {self.file_path}: {e}. Initializing new memory.")
