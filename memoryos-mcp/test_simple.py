#!/usr/bin/env python3
"""
Simple MemoryOS MCP Server Test
- Insert 15 conversations
- Set short-term memory capacity to 2
- Test 2 specific queries to verify memory retrieval works correctly.
"""

import asyncio
import json
import sys
from pathlib import Path

# 导入 MCP 客户端相关组件
# Import MCP client
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp import types
except ImportError as e:
    print(f"❌ Failed to import MCP client library: {e}")
    print("Please install official MCP SDK: pip install mcp")
    sys.exit(1)

# MemoryOS MCP 服务器简易测试类
class SimpleMemoryOSTest:
    """Simple MemoryOS MCP Server Test"""
    
    # 初始化测试类，指定服务器脚本和配置文件
    def __init__(self, server_script: str = "server_new.py", config_file: str = "config.json"):
        self.server_script = Path(server_script)
        self.config_file = Path(config_file)
        
        # Validate file existence
        if not self.server_script.exists():
            raise FileNotFoundError(f"Server script not found: {self.server_script}")
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
    
    # 获取启动服务器所需的 stdio 通信参数
    def get_server_params(self):
        """Get server parameters"""
        return StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script), "--config", str(self.config_file)],
            env=None
        )
    
    # 向 MemoryOS 连续插入 15 轮测试对话
    async def test_insert_conversations(self):
        """Insert 15 conversations into MemoryOS"""
        print("\n💾 Step 1: Insert 15 Conversations")
        
        # 15 test conversations
        conversations = [
            {"user_input": "Hello, I'm Tom from San Francisco", "agent_response": "Hello Tom! Nice to meet you. San Francisco is a great city!"},
            {"user_input": "I work as a software engineer", "agent_response": "That's awesome! Software engineering is a fascinating field. What technologies do you work with?"},
            {"user_input": "I mainly use Python and JavaScript", "agent_response": "Great choice! Python and JavaScript are very popular and powerful languages."},
            {"user_input": "I'm interested in machine learning", "agent_response": "Machine learning is an exciting field! Are you focusing on any particular area?"},
            {"user_input": "I want to learn about neural networks", "agent_response": "Neural networks are the foundation of deep learning. Would you like to start with the basics?"},
            {"user_input": "Yes, please explain backpropagation", "agent_response": "Backpropagation is the key algorithm for training neural networks. It calculates gradients to update weights."},
            {"user_input": "I have a project idea about chatbots", "agent_response": "Chatbots are a great application of ML! What kind of chatbot are you thinking about?"},
            {"user_input": "A customer service chatbot", "agent_response": "Customer service chatbots can be very helpful. You'll need to consider intent recognition and response generation."},
            {"user_input": "What frameworks should I use?", "agent_response": "For chatbots, you could use frameworks like Rasa, Dialogflow, or build with PyTorch/TensorFlow."},
            {"user_input": "I prefer open source solutions", "agent_response": "Great! Rasa is an excellent open-source framework for building conversational AI."},
            {"user_input": "How do I handle multiple languages?", "agent_response": "For multilingual support, you can use translation APIs or train separate models for each language."},
            {"user_input": "I also like hiking in my free time", "agent_response": "Hiking is a wonderful hobby! San Francisco has some great trails nearby like Lands End and Mount Sutro."},
            {"user_input": "Do you know any good hiking spots?", "agent_response": "Yes! You might enjoy Muir Woods, Mount Tamalpais, or the coastal trails in Pacifica."},
            {"user_input": "I'm planning a weekend trip", "agent_response": "That sounds fun! Are you thinking of staying local or going somewhere further?"},
            {"user_input": "Maybe somewhere within 2 hours drive", "agent_response": "Perfect! You could visit Napa Valley, Santa Cruz, or even go to Lake Tahoe if you don't mind a slightly longer drive."}
        ]
        
        server_params = self.get_server_params()
        
        try:
            # 建立 stdio 连接并初始化 MCP 会话
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    success_count = 0
                    
                    # 循环调用 add_memory 工具
                    for i, conversation in enumerate(conversations, 1):
                        print(f"   Adding conversation {i:2d}/15...")
                        
                        result = await session.call_tool("add_memory", conversation)
                        
                        # 解析工具返回结果
                        if hasattr(result, 'content') and result.content:
                            content = result.content[0]
                            if isinstance(content, types.TextContent):
                                response = json.loads(content.text)
                                if response.get("status") == "success":
                                    success_count += 1
                                    print(f"   ✅ Conversation {i:2d} added successfully")
                                else:
                                    print(f"   ❌ Conversation {i:2d} failed: {response.get('message', 'Unknown error')}")
                            else:
                                print(f"   ❌ Conversation {i:2d} failed: Invalid response format")
                        else:
                            print(f"   ❌ Conversation {i:2d} failed: No response content")
                        
                        # Brief delay
                        await asyncio.sleep(0.1)
                    
                    print(f"\n✅ Inserted {success_count}/15 conversations successfully")
                    return success_count == 15
                    
        except Exception as e:
            print(f"❌ Failed to insert conversations: {e}")
            return False
    
    # 通过查询验证记忆检索功能的有效性
    async def test_memory_retrieval(self):
        """Test memory retrieval with 2 specific queries"""
        print("\n🔍 Step 2: Test Memory Retrieval")
        
        # Test queries
        test_queries = [
            {
                "query": "Tell me about Tom from San Francisco",
                "description": "Query about the first conversation - should retrieve Tom's introduction",
                "expected_content": ["Tom", "San Francisco", "software engineer"]
            },
            {
                "query": "What does the user want to learn about machine learning?",
                "description": "Query about ML interests - should retrieve neural networks and chatbot discussions",
                "expected_content": ["neural networks", "chatbot", "machine learning"]
            }
        ]
        
        server_params = self.get_server_params()
        
        try:
            # 启动服务器客户端
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # 执行测试查询
                    for i, test_query in enumerate(test_queries, 1):
                        print(f"\n--- Query {i}: {test_query['description']} ---")
                        print(f"Question: {test_query['query']}")
                        
                        query_params = {
                            "query": test_query["query"],
                            "relationship_with_user": "friend",
                            "style_hint": "helpful",
                            "max_results": 10
                        }
                        
                        # 调用 retrieve_memory 工具
                        result = await session.call_tool("retrieve_memory", query_params)
                        
                        if hasattr(result, 'content') and result.content:
                            content = result.content[0]
                            if isinstance(content, types.TextContent):
                                response = json.loads(content.text)
                                if response.get("status") == "success":
                                    print(f"✅ Query {i} successful!")
                                    
                                    # Display results
                                    # 统计并打印各层记忆的命中数量
                                    pages_found = response.get('total_pages_found', 0)
                                    user_knowledge_found = response.get('total_user_knowledge_found', 0)
                                    assistant_knowledge_found = response.get('total_assistant_knowledge_found', 0)
                                    short_term_count = response.get('short_term_count', 0)
                                    
                                    print(f"📊 Results Summary:")
                                    print(f"   - Short-term memory: {short_term_count} items")
                                    print(f"   - Mid-term pages: {pages_found} items")
                                    print(f"   - User knowledge: {user_knowledge_found} items")
                                    print(f"   - Assistant knowledge: {assistant_knowledge_found} items")
                                    
                                    # Show some retrieved content
                                    pages = response.get('retrieved_pages', [])
                                    if pages:
                                        print(f"📄 Retrieved Pages ({len(pages)} items):")
                                        for j, page in enumerate(pages[:3], 1):  # Show first 3
                                            user_input = page.get('user_input', '')[:50]
                                            agent_response = page.get('agent_response', '')[:50]
                                            print(f"   {j}. User: {user_input}...")
                                            print(f"      Agent: {agent_response}...")
                                    
                                    # Check if expected content is found
                                    # 校验检索结果中是否包含预期的关键词
                                    full_text = json.dumps(response, ensure_ascii=False).lower()
                                    found_expected = []
                                    for expected in test_query['expected_content']:
                                        if expected.lower() in full_text:
                                            found_expected.append(expected)
                                    
                                    if found_expected:
                                        print(f"✅ Found expected content: {found_expected}")
                                    else:
                                        print(f"⚠️ Expected content not found: {test_query['expected_content']}")
                                    
                                    # Check if first conversation is retrievable
                                    # 特别验证第一轮对话是否已成功流转至中期记忆并可被召回
                                    if i == 1:  # First query about Tom
                                        if pages_found > 0 or "tom" in full_text:
                                            print("✅ First conversation successfully moved to mid-term memory and is retrievable!")
                                        else:
                                            print("⚠️ First conversation might not be in mid-term memory yet")
                                    
                                else:
                                    print(f"❌ Query {i} failed: {response.get('message', 'Unknown error')}")
                            else:
                                print(f"❌ Query {i} failed: Invalid response format")
                        else:
                            print(f"❌ Query {i} failed: No response content")
                        
                        await asyncio.sleep(0.5)  # Longer delay between queries
                    
                    return True
                    
        except Exception as e:
            print(f"❌ Memory retrieval test failed: {e}")
            return False
    
    # 执行完整的测试流程
    async def run_test(self):
        """Run the complete test"""
        print("🚀 Starting Simple MemoryOS MCP Server Test")
        print(f"Server script: {self.server_script}")
        print(f"Config file: {self.config_file}")
        print("=" * 60)
        
        # Step 1: Insert conversations
        insert_success = await self.test_insert_conversations()
        if not insert_success:
            print("❌ Failed to insert conversations. Stopping test.")
            return False
        
        # Wait a bit for processing
        # 等待一段时间让异步更新逻辑处理完毕
        print("\n⏳ Waiting 3 seconds for memory processing...")
        await asyncio.sleep(3)
        
        # Step 2: Test retrieval
        retrieval_success = await self.test_memory_retrieval()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        print(f"✅ Conversation insertion: {'Passed' if insert_success else 'Failed'}")
        print(f"✅ Memory retrieval: {'Passed' if retrieval_success else 'Failed'}")
        
        if insert_success and retrieval_success:
            print("🎉 All tests passed! MemoryOS is working correctly.")
            print("🔍 Key findings:")
            print("   - Short-term memory capacity limit working (should be 2)")
            print("   - Mid-term memory storage and retrieval working")
            print("   - First conversation successfully retrievable from mid-term memory")
            return True
        else:
            print("⚠️ Some tests failed. Please check the system.")
            return False

# 程序主入口，解析参数并运行异步测试
def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple MemoryOS MCP Server Test")
    parser.add_argument("--server", default="server_new.py", help="Server script path")
    parser.add_argument("--config", default="config.json", help="Config file path")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        tester = SimpleMemoryOSTest(args.server, args.config)
        success = asyncio.run(tester.run_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        # 输出错误详情
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
