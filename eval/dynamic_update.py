from utils import gpt_summarize, generate_id, get_timestamp, gpt_update_profile, gpt_generate_multi_summary

# 动态更新器类，负责短期记忆向中、长期记忆的平滑流转
class DynamicUpdate:
    # 初始化动态更新器，挂载各类存储模块和 LLM 客户端
    def __init__(self, short_term_memory, mid_term_memory, long_term_memory, topic_similarity_threshold=0.8, client=None):
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.topic_similarity_threshold = topic_similarity_threshold
        self.client = client
        # 跟踪上一个被驱逐的页面，用于维持对话链条
        self.last_evicted_page = None

    # 调用 LLM 判断两页对话之间是否具有连贯的话题
    def _is_conversation_continuing(self, previous_page, current_page):
        if not previous_page:
            return False
            
        prompt = """Determine if these two conversation pages are continuous (true continuation without topic shift).
Return ONLY "true" or "false".

Previous Page:
User: {prev_user}
Assistant: {prev_agent}

Current Page:
User: {curr_user}
Assistant: {curr_agent}

Continuous?""".format(
            prev_user=previous_page.get("user_input", ""),
            prev_agent=previous_page.get("agent_response", ""),
            # 当前轮次的输入与响应
            curr_user=current_page.get("user_input", ""),
            curr_agent=current_page.get("agent_response", "")
        )
        
        messages = [
            {"role": "system", "content": "You are a conversation continuity detector. Return ONLY 'true' or 'false'."},
            {"role": "user", "content": prompt}
        ]
        
        # 采用 0.0 温度以获得稳定的逻辑判断
        response = self.client.chat_completion(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=10
        )
        
        return response.strip().lower() == "true"

    # 根据上下文更新对话的元级描述信息
    def _generate_meta_info(self, last_page_meta, current_page):
        """
        基于上一页的meta-info和当前页内容生成新的meta-info
        :param last_page_meta: 上一页的meta-info内容
        :param current_page: 当前页的对话内容
        :return: 更新后的meta-info
        """
        current_conversation = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
        
        prompt = """Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.
        
    Guidelines:
    1. Start from the previous meta-summary (if exists)
    2. Add/update information based on the new dialogue
    3. Keep it concise (1-2 sentences max)
    4. Maintain context coherence

    Previous Meta-summary: {last_meta}
    New Dialogue:
    {new_dialogue}

    Updated Meta-summary:""".format(
            last_meta=last_page_meta if last_page_meta else "None",
            new_dialogue=current_conversation
        )
        
        messages = [
            {"role": "system", "content": """You are a conversation meta-summary updater. Your task is to:
    1. Preserve relevant context from previous meta-summary
    2. Integrate new information from current dialogue
    3. Output ONLY the updated summary (no explanations)"""},
            {"role": "user", "content": prompt}
        ]
        
        return self.client.chat_completion(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=100
        ).strip()

    # 传播元信息更新到与其关联的所有相邻页面
    def _update_connected_pages(self, page_id, new_meta_info):
        connected_pages = []
        current_page = self.mid_term_memory.get_page_by_id(page_id)
        
        if not current_page:
            return
            
        # 向前回溯所有关联页面
        prev_page_id = current_page.get("pre_page")
        while prev_page_id:
            prev_page = self.mid_term_memory.get_page_by_id(prev_page_id)
            if prev_page:
                connected_pages.insert(0, prev_page)
                prev_page_id = prev_page.get("pre_page")
            else:
                break
                
        # 向后追踪所有后续关联页面
        next_page_id = current_page.get("next_page")
        while next_page_id:
            next_page = self.mid_term_memory.get_page_by_id(next_page_id)
            if next_page:
                connected_pages.append(next_page)
                next_page_id = next_page.get("next_page")
            else:
                break
                
        # 统一刷新该逻辑链条中的元摘要描述
        for page in connected_pages:
            page["meta_info"] = new_meta_info
            self.mid_term_memory.update_page_connections(page.get("pre_page"), page.get("next_page"))

    # 将新消息暂存至短期记忆
    def update_short_term(self, message):
        self.short_term_memory.add_qa_pair(message)

    # 批量从短期记忆淘汰并迁移至中期记忆的核心引擎
    def bulk_evict_and_update_mid_term(self):
        evicted = []
        # 1. 从短期记忆移除内容（保持不变）
        # 持续收集超限的旧对话
        while self.short_term_memory.is_full():
            msg = self.short_term_memory.pop_oldest()
            if msg and msg.get("user_input") and msg.get("agent_response"):
                evicted.append(msg)
        
        if not evicted:
            return
        
        # 2. 先创建基础页面结构并进行连续性处理
        # 构建记忆页面，并调用 LLM 建立对话链
        pages = []
        for qa in evicted:
            page = {
                "page_id": generate_id("page"),
                "user_input": qa.get("user_input", ""),
                "agent_response": qa.get("agent_response", ""),
                "timestamp": qa.get("timestamp"),
                "preloaded": False,
                "analyzed": False,
                "pre_page": None,
                "next_page": None,
                "meta_info": None
            }
            
            # 连续性判断
            # 检查是否是上一次对话的延续
            is_continuous = self._is_conversation_continuing(self.last_evicted_page, page)
            if is_continuous and self.last_evicted_page:
                page["pre_page"] = self.last_evicted_page["page_id"]
                self.last_evicted_page["next_page"] = page["page_id"]
                
                # 更新元信息
                # 生成新的对话链元摘要并广播到整条链
                last_meta = self.last_evicted_page.get("meta_info")
                new_meta_info = self._generate_meta_info(last_meta, page)
                page["meta_info"] = new_meta_info
                self._update_connected_pages(page["pre_page"], new_meta_info)
            else:
                # 作为新对话链条的起始
                page["meta_info"] = self._generate_meta_info(None, page)
            
            pages.append(page)
            self.last_evicted_page = page
        
        # 3. 将所有用户输入拼接用于主题分析
        # 通过拼接文本，让 LLM 识别当前批次的子话题
        input_text = "\n".join([f"User: {page.get('user_input','')}\n" for page in pages])
        print("动态更新：调用 GPT 生成多子主题摘要...")
        multi_summary = gpt_generate_multi_summary(input_text, self.client)
        
        # 4. 按主题分组插入中期记忆
        # 将对话页面按照识别出的主题聚类存入中期存储
        for summary_dict in multi_summary.get("summaries", []):
            sub_summary = summary_dict.get("content", "")
            sub_key_words = summary_dict.get("keywords", [])
            
            print(f"动态更新：处理子主题【{summary_dict.get('theme','')}】，插入中期记忆...")
            self.mid_term_memory.insert_pages_into_session(
                sub_summary, 
                sub_key_words, 
                pages,  # 传入已经处理好的完整pages
                self.topic_similarity_threshold
            )

    # 根据深度分析结果更新长期画像及私有知识库
    def update_long_term(self, user_id, new_profile_data, knowledge_text):
        print("动态更新：更新长期记忆中的用户画像和私有数据...")
        self.long_term_memory.update_user_profile(user_id, new_profile_data)
        self.long_term_memory.add_knowledge(knowledge_text)
