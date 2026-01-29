
import os
from memoryos import Memoryos

# --- Basic Configuration ---
# 基础配置：用户ID、助手ID、API密钥等
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
API_KEY = ""  # Replace with your key
BASE_URL = ""  # Optional: if using a custom OpenAI endpoint
DATA_STORAGE_PATH = ""
LLM_MODEL = "gpt-4o-mini"

# 简单的演示函数，展示 MemoryOS 的基本用法
def simple_demo():
    print("MemoryOS Simple Demo")
    
    # 1. Initialize MemoryOS
    # 1. 初始化 MemoryOS 实例
    print("Initializing MemoryOS...")
    try:
        memo = Memoryos(
            user_id=USER_ID,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=DATA_STORAGE_PATH,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,  
            mid_term_heat_threshold=1000,  
            retrieval_queue_capacity=10,
            long_term_knowledge_capacity=100,
            mid_term_similarity_threshold=0.6,
            embedding_model_name=""
        )
        print("MemoryOS initialized successfully!\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Add some basic memories
    # 2. 添加一些基础记忆（对话对）
    print("Adding some memories...")
    
    # 添加 Tom 的个人职业信息
    memo.add_memory(
        user_input="Hi! I'm Tom, I work as a data scientist in San Francisco.",
        agent_response="Hello Tom! Nice to meet you. Data science is such an exciting field. What kind of data do you work with?"
    )
    # 添加兴趣爱好：徒步
    memo.add_memory(
        user_input="I love hiking on weekends, especially in the mountains.",
        agent_response="That sounds wonderful! Do you have a favorite trail or mountain you like to visit?"
    )
    # 添加兴趣爱好：AI
    memo.add_memory(
        user_input="Recently, I've been reading a lot about artificial intelligence.",
        agent_response="AI is a fascinating topic! Are you interested in any specific area of AI?"
    )
    # 添加食物偏好：寿司
    memo.add_memory(
        user_input="My favorite food is sushi, especially salmon nigiri.",
        agent_response="Sushi is delicious! Have you ever tried making it at home?"
    )
    # 添加宠物信息：金毛犬 Max
    memo.add_memory(
        user_input="I have a golden retriever named Max.",
        agent_response="Max must be adorable! How old is he?"
    )
    # 添加旅行经历：日本
    memo.add_memory(
        user_input="I traveled to Japan last year and visited Tokyo and Kyoto.",
        agent_response="That must have been an amazing experience! What did you enjoy most about Japan?"
    )
    # 添加学习技能：吉他
    memo.add_memory(
        user_input="I'm currently learning how to play the guitar.",
        agent_response="That's awesome! What songs are you practicing right now?"
    )
    # 添加生活习惯：咖啡
    memo.add_memory(
        user_input="I usually start my day with a cup of black coffee.",
        agent_response="Coffee is a great way to kickstart the day! Do you prefer it hot or iced?"
    )
    # 添加电影偏好：科幻
    memo.add_memory(
        user_input="My favorite movie genre is science fiction.",
        agent_response="Sci-fi movies can be so imaginative! Do you have a favorite film?"
    )
    # 添加兴趣爱好：绘画
    memo.add_memory(
        user_input="I enjoy painting landscapes in my free time.",
        agent_response="Painting is such a creative hobby! Do you use oils, acrylics, or watercolors?"
    )

    # 测试查询，验证系统是否记得之前提到的工作信息

    test_query = "What do you remember about my job?"
    print(f"User: {test_query}")
    
    # 获取响应
    response = memo.get_response(
        query=test_query,
    )
    
    print(f"Assistant: {response}")

# 脚本入口
if __name__ == "__main__":
    simple_demo()
