import json
import numpy as np
from utils import get_timestamp, get_embedding, normalize_vector

# 长期记忆类，管理用户画像和私有知识库
class LongTermMemory:
    # 初始化长期记忆，配置存储文件并加载历史数据
    def __init__(self, file_path="long_term.json"):
        self.file_path = file_path
        self.user_profiles = {}
        self.knowledge_base = []
        self.assistant_knowledge = []
        self.load()

    # 更新用户画像信息，支持追加或直接覆盖模式
    def update_user_profile(self, user_id, new_data, merge=False):
        """
        更新用户画像
        :param user_id: 用户ID
        :param new_data: 新数据
        :param merge: 是否合并到现有数据 (True) 或覆盖 (False)
        """
        if merge and user_id in self.user_profiles:
            current_data = self.user_profiles[user_id]["data"]
            if isinstance(current_data, str) and isinstance(new_data, str):
                # 如果是文本格式的画像，保留原有数据并追加新数据
                updated_data = f"{current_data}\n\n--- Updated ---\n{new_data}"
            else:
                updated_data = new_data  # 如果不是字符串，直接覆盖(需要更复杂的合并逻辑)
        else:
            updated_data = new_data
        
        # 记录画像内容及最后更新时间
        self.user_profiles[user_id] = {
            "data": updated_data,
            "last_updated": get_timestamp()
        }
        print("长期记忆：更新用户画像。")
        self.save()

    # 添加关于助手的知识或身份特征条目
    def add_assistant_knowledge(self, knowledge_text):
        """
        添加助手相关的知识或特性
        """
        # 过滤无效的空文本或占位符
        if knowledge_text.strip() == "" or knowledge_text.strip() == "- None" or knowledge_text.strip() == "- None.":
            print("长期记忆：助手知识为空，不保存。")
            return
        # 生成并存储知识向量，用于后续相似度检索
        vec = get_embedding(knowledge_text)
        vec = normalize_vector(vec).tolist()
        entry = {
            "knowledge": knowledge_text,
            "timestamp": get_timestamp(),
            "knowledge_embedding": vec
        }
        self.assistant_knowledge.append(entry)
        print("长期记忆：添加助手知识。")
        self.save()

    # 获取当前加载的所有助手知识列表
    def get_assistant_knowledge(self):
        """
        获取所有助手知识
        """
        return self.assistant_knowledge


    # 获取指定用户的原始画像字符串内容
    def get_raw_user_profile(self, user_id):
        """获取原始用户画像数据"""
        return self.user_profiles.get(user_id, {}).get("data", "")
    
    # 获取包含元数据的完整用户画像字典
    def get_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {})

    # 添加用户的私有知识条目
    def add_knowledge(self, knowledge_text):
        # 过滤无效输入
        if knowledge_text.strip() == "" or knowledge_text.strip() == "- None"or knowledge_text.strip() == "- None.":
            print("长期记忆：私有知识为空，不保存。")
            return
        # 计算并保存嵌入向量
        vec = get_embedding(knowledge_text)
        vec = normalize_vector(vec).tolist()
        entry = {
            "knowledge": knowledge_text,
            "timestamp": get_timestamp(),
            "knowledge_embedding": vec
        }
        self.knowledge_base.append(entry)
        print("长期记忆：添加私有知识。")
        self.save()

    # 获取当前加载的所有用户知识库列表
    def get_knowledge(self):
        return self.knowledge_base

    # 在用户知识库中执行向量相似度搜索
    def search_knowledge(self, query, threshold=0.1, top_k=10):
        if not self.knowledge_base:
            return []
        # 计算查询向量并进行归一化
        query_vec = get_embedding(query)
        query_vec = normalize_vector(query_vec)
        embeddings = []
        for entry in self.knowledge_base:
            embeddings.append(np.array(entry["knowledge_embedding"], dtype=np.float32))
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        # 利用 FAISS 的内积索引进行高效召回
        from faiss import IndexFlatIP
        dim = embeddings.shape[1]
        index = IndexFlatIP(dim)
        index.add(embeddings)
        query_arr = np.array([query_vec], dtype=np.float32)
        # 搜索最相关的 top_k 条知识
        distances, indices = index.search(query_arr, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # 仅保留相似度超过预设阈值的结果
            if dist >= threshold:
                results.append(self.knowledge_base[idx])
        print(f"长期记忆：检索到 {len(results)} 个匹配知识。")
        return results

    # 持久化所有画像和知识库数据到 JSON
    def save(self):
        data = {
            "user_profiles": self.user_profiles,
            "knowledge_base": self.knowledge_base,
            "assistant_knowledge": self.assistant_knowledge
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("长期记忆：保存成功。")

    # 从磁盘文件恢复长期记忆数据结构
    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.user_profiles = data.get("user_profiles", {})
                self.knowledge_base = data.get("knowledge_base", [])
                self.assistant_knowledge = data.get("assistant_knowledge", [])  # 加载助手知识
            print("长期记忆：加载成功。")
        except Exception:
            # 加载失败则初始化为空容器
            self.user_profiles = {}
            self.knowledge_base = []
            print("长期记忆：无历史数据。")
