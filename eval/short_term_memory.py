import json
from collections import deque
from utils import get_timestamp

# 短期记忆类，用于管理最近轮次的对话对
class ShortTermMemory:
    # 初始化短期记忆，设置最大容量和文件存储路径
    def __init__(self, max_capacity=10, file_path="short_term.json"):
        self.max_capacity = max_capacity
        self.file_path = file_path
        # 使用双端队列存储记忆，达到容量后自动挤出旧项
        self.memory = deque(maxlen=max_capacity)
        # 启动时加载现有数据
        self.load()

    # 向短期记忆中添加一个新的 QA 问答对
    def add_qa_pair(self, qa_pair):
        # 如果 QA 对中没有时间戳，则自动获取当前时间
        qa_pair["timestamp"] = qa_pair.get("timestamp", get_timestamp())
        self.memory.append(qa_pair)
        print(f"短期记忆：添加 QA 对，用户: {qa_pair.get('user_input','')[:30]}...")
        # 持久化保存到磁盘
        self.save()

    # 获取当前内存中所有的 QA 对
    def get_all(self):
        return list(self.memory)

    # 检查短期记忆是否已达到最大预设容量
    def is_full(self):
        return len(self.memory) == self.max_capacity

    # 手动弹出并移除最久远的一个问答对
    def pop_oldest(self):
        if self.memory:
            msg = self.memory.popleft()
            print("短期记忆：淘汰最老 QA 对。")
            # 移除后更新磁盘文件
            self.save()
            return msg
        return None

    # 将当前短期记忆对象序列化保存为 JSON 文件
    def save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(list(self.memory), f, ensure_ascii=False, indent=2)

    # 从磁盘 JSON 文件中读取并恢复短期记忆数据
    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.memory = deque(data, maxlen=self.max_capacity)
            print("短期记忆：加载成功。")
        except Exception:
            # 如果加载失败或文件不存在，则初始化为空队列
            self.memory = deque(maxlen=self.max_capacity)
            print("短期记忆：无历史数据。")
