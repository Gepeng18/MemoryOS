import json
from collections import deque
try:
    # 尝试相对导入
    from .utils import get_timestamp, ensure_directory_exists
except ImportError:
    # 回退到绝对导入
    from utils import get_timestamp, ensure_directory_exists

# 短期记忆类，用于管理最近的问答对交互
class ShortTermMemory:
    # 初始化短期记忆，设置最大容量和文件路径
    def __init__(self, file_path, max_capacity=10):
        self.max_capacity = max_capacity
        self.file_path = file_path
        # 确保存储目录存在
        ensure_directory_exists(self.file_path)
        # 使用双端队列存储记忆，达到最大容量时自动移除最旧项
        self.memory = deque(maxlen=max_capacity)
        # 从磁盘加载现有记忆
        self.load()

    # 添加新的问答对到短期记忆
    def add_qa_pair(self, qa_pair):
        # 确保存在时间戳，如果没有则添加
        # Ensure timestamp exists, add if not
        if 'timestamp' not in qa_pair or not qa_pair['timestamp']:
            qa_pair["timestamp"] = get_timestamp()
        
        self.memory.append(qa_pair)
        print(f"ShortTermMemory: Added QA. User: {qa_pair.get('user_input','')[:30]}...")
        # 保存更新后的记忆到磁盘
        self.save()

    # 获取所有短期记忆项
    def get_all(self):
        return list(self.memory)

    # 检查短期记忆是否已达到最大容量
    def is_full(self):
        # 使用 >= 以确保安全
        return len(self.memory) >= self.max_capacity # Use >= to be safe

    # 弹出并返回最旧的问答对
    def pop_oldest(self):
        if self.memory:
            msg = self.memory.popleft()
            print("ShortTermMemory: Evicted oldest QA pair.")
            # 移除后保存记忆
            self.save()
            return msg
        return None

    # 将当前短期记忆保存到 JSON 文件
    def save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(list(self.memory), f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving ShortTermMemory to {self.file_path}: {e}")

    # 从 JSON 文件加载短期记忆
    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 确保数据加载正确，处理文件为空或格式错误的情况
                # Ensure items are loaded correctly, especially if file was empty or malformed
                if isinstance(data, list):
                    self.memory = deque(data, maxlen=self.max_capacity)
                else:
                    self.memory = deque(maxlen=self.max_capacity)
            print(f"ShortTermMemory: Loaded from {self.file_path}.")
        except FileNotFoundError:
            self.memory = deque(maxlen=self.max_capacity)
            print(f"ShortTermMemory: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            self.memory = deque(maxlen=self.max_capacity)
            print(f"ShortTermMemory: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            self.memory = deque(maxlen=self.max_capacity)
            # 处理加载过程中发生的非预期错误
            print(f"ShortTermMemory: An unexpected error occurred during load from {self.file_path}: {e}. Initializing new memory.")
