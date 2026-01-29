"""
This file stores all the prompts used by the Memoryos system.
"""

# 生成系统响应的提示词（系统部分）
# 翻译：作为一名具有优秀沟通习惯的沟通专家，你在以下对话中扮演{relationship}的角色。
# 这里是你的一些独特个性特征和知识：{assistant_knowledge_text}
# 用户画像：{meta_data_text}
# 你的任务是生成符合这些特征并保持语气的回复。
# Prompt for generating system response (from main_memoybank.py, generate_system_response_with_meta)
GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT = (
    "As a communication expert with outstanding communication habits, you embody the role of {relationship} throughout the following dialogues.\n"
    "Here are some of your distinctive personal traits and knowledge:\n{assistant_knowledge_text}\n"
    "User's profile:\n"
    "{meta_data_text}\n"
    "Your task is to generate responses that align with these traits and maintain the tone.\n"
)

# 生成系统响应的提示词（用户部分）
# 翻译：<上下文> 根据你最近与用户的对话：{history_text}
# <记忆> 与正在进行的对话相关的记忆是：{retrieval_text}
# <用户特征> 在过去你与用户的对话过程中，你发现用户具有以下特征：{background}
# 现在，请扮演{relationship}继续你与用户之间的对话。
# 用户刚刚说：{query}
# 请使用以下格式回答用户的陈述（最多30个词，必须使用英文）：
# 在回答问题时，请务必检查所引用信息的时间戳是否与问题的时间范围匹配。
GENERATE_SYSTEM_RESPONSE_USER_PROMPT = (
    "<CONTEXT>\n"
    "Drawing from your recent conversation with the user:\n"
    "{history_text}\n\n"
    "<MEMORY>\n"
    "The memories linked to the ongoing conversation are:\n"
    "{retrieval_text}\n\n"
    "<USER TRAITS>\n"
    "During the conversation process between you and the user in the past, you found that the user has the following characteristics:\n"
    "{background}\n\n"
    "Now, please role-play as {relationship} to continue the dialogue between you and the user.\n"
    "The user just said: {query}\n"
    "Please respond to the user's statement using the following format (maximum 30 words, must be in English):\n "
    "When answering questions, be sure to check whether the timestamp of the referenced information matches the timeframe of the question"
)

# 助手知识提取提示词（系统部分）
# 翻译：你是一个助手知识提取引擎。规则：
# 1. 仅提取关于助手身份或知识的明确陈述。
# 2. 使用第一人称进行简明、事实性的陈述。
# 3. 如果未发现相关信息，输出 "None"。
# Prompt for assistant knowledge extraction (from utils.py, analyze_assistant_knowledge)
ASSISTANT_KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None"."""

# 助手知识提取提示词（用户部分）
# 翻译：# 助手知识提取任务
# 分析对话并提取关于助手的任何事实或身份特征。
# 如果无法提取任何特征，请回复 "None"。使用以下格式输出：
# 生成的内容应尽可能简洁 —— 越简洁越好。
# 【助手知识】
# - [事实 1]
# - [事实 2]
# - (如果未发现则写 "None")
ASSISTANT_KNOWLEDGE_EXTRACTION_USER_PROMPT = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the assistant. 
If no traits can be extracted, reply with "None". Use the following format for output:
The generated content should be as concise as possible — the more concise, the better.
【Assistant Knowledge】
- [Fact 1]
- [Fact 2]
- (Or "None" if none found)

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   【Assistant Knowledge】
   - I recommend Interstellar on 2023-10-01.

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   【Assistant Knowledge】
   - I have cooking recipes and techniques on 2023-10-02.

3. User: That's interesting. I didn't know you could do that.
   AI: I'm glad you find it interesting!
   【Assistant Knowledge】
   - None

Conversation:
{conversation}
"""

# 对话摘要生成提示词
# 翻译：系统：你是一个总结对话话题的专家。生成极其简明且精确的摘要。尽可能简短，同时捕捉核心要点。
# 用户：请根据以下对话生成简明的话题摘要。最多保持在2-3个短句：{dialog_text} 简明摘要：
# Prompt for summarizing dialogs (from utils.py, gpt_summarize)
SUMMARIZE_DIALOGS_SYSTEM_PROMPT = "You are an expert in summarizing dialogue topics. Generate extremely concise and precise summaries. Be as brief as possible while capturing the essence."
SUMMARIZE_DIALOGS_USER_PROMPT = "Please generate an concise topic summary based on the following conversation. Keep it to 2-3 short sentences maximum:\n{dialog_text}\nConcise Summary："

# 多主题摘要生成提示词
# 翻译：系统：你是一个分析对话话题的专家。生成简明摘要。不要超过两个话题。尽可能简短。
# 用户：请分析以下对话并生成极其简明的子话题摘要（如果适用），最多包含两个主题。
# 每个摘要都应非常简短 —— 仅针对主题和内容使用几个词。格式化为 JSON 数组：
# [ { "theme": "简短主题", "keywords": ["关键词1", "关键词2"], "content": "摘要内容" } ]
# 对话内容：{text}
# Prompt for multi-summary generation (from utils.py, gpt_generate_multi_summary)
MULTI_SUMMARY_SYSTEM_PROMPT = "You are an expert in analyzing dialogue topics. Generate  concise summaries. No more than two topics. Be as brief as possible."
MULTI_SUMMARY_USER_PROMPT = ("Please analyze the following dialogue and generate extremely concise subtopic summaries (if applicable), with a maximum of two themes.\n"
                           "Each summary should be very brief - just a few words for the theme and content. Format as JSON array:\n"
                           "[\n  {{\"theme\": \"Brief theme\", \"keywords\": [\"key1\", \"key2\"], \"content\": \"summary\"}}\n]\n"
                           "\nConversation content:\n{text}")

# 性格分析提示词（系统部分）
# 翻译：你是一个专业的用户偏好分析助手。你的任务是从给定的对话中根据提供的维度分析用户的性格偏好。
# 对于每个维度：
# 1. 仔细阅读对话并确定是否反映了该维度。
# 2. 如果反映了，确定用户的偏好级别：高 / 中 / 低，并简要解释推理，尽可能包括时间、人物和背景。
# 3. 如果未反映该维度，请勿提取或列出它。
# 性格分析部分仅关注用户的偏好和特征。仅输出用户画像部分。
# Prompt for personality analysis (NEW TEMPLATE)
PERSONALITY_ANALYSIS_SYSTEM_PROMPT = """You are a professional user preference analysis assistant. Your task is to analyze the user's personality preferences from the given dialogue based on the provided dimensions.

For each dimension:
1. Carefully read the conversation and determine if the dimension is reflected.
2. If reflected, determine the user's preference level: High / Medium / Low, and briefly explain the reasoning, including time, people, and context if possible.
3. If the dimension is not reflected, do not extract or list it.

Focus only on the user's preferences and traits for the personality analysis section.
Output only the user profile section.
"""

# 性格分析提示词（用户部分）
# 翻译：请分析下面的最新用户与 AI 对话，并根据 90 个性格偏好维度更新用户画像。
# 这里是 90 个维度及其解释：... (省略详细维度翻译) ...
# 任务说明：
# 1. 查看下方的现有用户画像。
# 2. 分析新对话中关于上述 90 个维度的证据。
# 3. 更新并将发现整合到一个全面的用户画像中。
# 4. 对于可以识别的每个维度，使用格式：维度 ( 级别(高/中/低) )。
# 5. 尽可能为每个维度提供简短的推理。
# 6. 保留旧画像中的现有见解，同时整合新的观察结果。
# 7. 如果无法从旧画像或新对话中推断出某个维度，请勿包含它。
PERSONALITY_ANALYSIS_USER_PROMPT = """Please analyze the latest user-AI conversation below and update the user profile based on the 90 personality preference dimensions.

Here are the 90 dimensions and their explanations:

[Psychological Model (Basic Needs & Personality)]
Extraversion: Preference for social activities.
Openness: Willingness to embrace new ideas and experiences.
Agreeableness: Tendency to be friendly and cooperative.
Conscientiousness: Responsibility and organizational ability.
Neuroticism: Emotional stability and sensitivity.
Physiological Needs: Concern for comfort and basic needs.
Need for Security: Emphasis on safety and stability.
Need for Belonging: Desire for group affiliation.
Need for Self-Esteem: Need for respect and recognition.
Cognitive Needs: Desire for knowledge and understanding.
Aesthetic Appreciation: Appreciation for beauty and art.
Self-Actualization: Pursuit of one's full potential.
Need for Order: Preference for cleanliness and organization.
Need for Autonomy: Preference for independent decision-making and action.
Need for Power: Desire to influence or control others.
Need for Achievement: Value placed on accomplishments.

[AI Alignment Dimensions]
Helpfulness: Whether the AI's response is practically useful to the user. (This reflects user's expectation of AI)
Honesty: Whether the AI's response is truthful. (This reflects user's expectation of AI)
Safety: Avoidance of sensitive or harmful content. (This reflects user's expectation of AI)
Instruction Compliance: Strict adherence to user instructions. (This reflects user's expectation of AI)
Truthfulness: Accuracy and authenticity of content. (This reflects user's expectation of AI)
Coherence: Clarity and logical consistency of expression. (This reflects user's expectation of AI)
Complexity: Preference for detailed and complex information.
Conciseness: Preference for brief and clear responses.

[Content Platform Interest Tags]
Science Interest: Interest in science topics.
Education Interest: Concern with education and learning.
Psychology Interest: Interest in psychology topics.
Family Concern: Interest in family and parenting.
Fashion Interest: Interest in fashion topics.
Art Interest: Engagement with or interest in art.
Health Concern: Concern with physical health and lifestyle.
Financial Management Interest: Interest in finance and budgeting.
Sports Interest: Interest in sports and physical activity.
Food Interest: Passion for cooking and cuisine.
Travel Interest: Interest in traveling and exploring new places.
Music Interest: Interest in music appreciation or creation.
Literature Interest: Interest in literature and reading.
Film Interest: Interest in movies and cinema.
Social Media Activity: Frequency and engagement with social media.
Tech Interest: Interest in technology and innovation.
Environmental Concern: Attention to environmental and sustainability issues.
History Interest: Interest in historical knowledge and topics.
Political Concern: Interest in political and social issues.
Religious Interest: Interest in religion and spirituality.
Gaming Interest: Enjoyment of video games or board games.
Animal Concern: Concern for animals or pets.
Emotional Expression: Preference for direct vs. restrained emotional expression.
Sense of Humor: Preference for humorous or serious communication style.
Information Density: Preference for detailed vs. concise information.
Language Style: Preference for formal vs. casual tone.
Practicality: Preference for practical advice vs. theoretical discussion.

**Task Instructions:**
1. Review the existing user profile below
2. Analyze the new conversation for evidence of the 90 dimensions above
3. Update and integrate the findings into a comprehensive user profile
4. For each dimension that can be identified, use the format: Dimension ( Level(High/Medium/Low) )
5. Include brief reasoning for each dimension when possible
6. Maintain existing insights from the old profile while incorporating new observations
7. If a dimension cannot be inferred from either the old profile or new conversation, do not include it

**Existing User Profile:**
{existing_user_profile}

**Latest User-AI Conversation:**
{conversation}

**Updated User Profile:**
Please provide the comprehensive updated user profile below, combining insights from both the existing profile and new conversation:"""

# 知识提取提示词（系统部分）
# 翻译：你是一个知识提取助手。你的任务是从对话中提取用户隐私数据和助手知识。
# 重点关注：
# 1. 用户隐私数据：个人信息、偏好或关于用户的私人事实。
# 2. 助手知识：关于助手做了什么、提供了什么或演示了什么的明确陈述。
# 提取时请做到极度简明且符合事实。使用尽可能短的短语。
# Prompt for knowledge extraction (NEW)
KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction assistant. Your task is to extract user private data and assistant knowledge from conversations.

Focus on:
1. User private data: personal information, preferences, or private facts about the user
2. Assistant knowledge: explicit statements about what the assistant did, provided, or demonstrated

Be extremely concise and factual in your extractions. Use the shortest possible phrases.
"""

# 知识提取提示词（用户部分）
# 翻译：请从下方的最新用户与 AI 对话中提取用户隐私数据和助手知识。
# 【用户隐私数据】 提取关于用户的个人信息。极度简明 —— 使用最短的短语：
# - [简短事实]: [最小上下文（包括实体和时间）]
# 【助手知识】 提取助手演示的内容。使用格式 "Assistant [动作] at [时间]"。极度简短。
KNOWLEDGE_EXTRACTION_USER_PROMPT = """Please extract user private data and assistant knowledge from the latest user-AI conversation below.

Latest User-AI Conversation:
{conversation}

【User Private Data】
Extract personal information about the user. Be extremely concise - use shortest possible phrases:
- [Brief fact]: [Minimal context(Including entities and time)]
- [Brief fact]: [Minimal context(Including entities and time)]
- (If no private data found, write "None")

【Assistant Knowledge】
Extract what the assistant demonstrated. Use format "Assistant [action] at [time]". Be extremely brief:
- Assistant [brief action] at [time/context]
- Assistant [brief capability] during [brief context]
- (If no assistant knowledge found, write "None")
"""

# 更新用户画像提示词
# 翻译：系统：你是一个合并和更新用户画像的专家。将新信息整合到旧画像中，保持一致性并提高对用户的整体理解。避免冗余。新的分析基于特定维度，尝试有意义地整合这些见解。
# 用户：请根据新的分析更新以下用户画像。如果旧画像为空或为 "None"，请根据新分析创建一个。
# 旧画像：{old_profile} 新分析数据：{new_analysis} 更新后的画像：
# Prompt for updating user profile (from utils.py, gpt_update_profile)
UPDATE_PROFILE_SYSTEM_PROMPT = "You are an expert in merging and updating user profiles. Integrate the new information into the old profile, maintaining consistency and improving the overall understanding of the user. Avoid redundancy. The new analysis is based on specific dimensions, try to incorporate these insights meaningfully."
UPDATE_PROFILE_USER_PROMPT = "Please update the following user profile based on the new analysis. If the old profile is empty or \"None\", create a new one based on the new analysis.\n\nOld User Profile:\n{old_profile}\n\nNew Analysis Data:\n{new_analysis}\n\nUpdated User Profile:"

# 提取主题提示词
# 翻译：系统：你是一个从文本中提取主旨的专家。提供一个简明的主题。
# 用户：请从以下文本中提取主旨：{answer_text} 主题：
# Prompt for extracting theme (from utils.py, gpt_extract_theme)
EXTRACT_THEME_SYSTEM_PROMPT = "You are an expert in extracting the main theme from a text. Provide a concise theme."
EXTRACT_THEME_USER_PROMPT = "Please extract the main theme from the following text:\n{answer_text}\n\nTheme:"



# 对话连贯性检查提示词
# 翻译：系统：你是一个对话连贯性检测器。仅返回 'true' 或 'false'。
# 用户：确定这两个对话页面是否连贯（在没有话题偏移的情况下真实延续）。仅返回 'true' 或 'false'。
# 前一页：... 当前页：... 是否连贯？
# Prompt for conversation continuity check (from dynamic_update.py, _is_conversation_continuing)
CONTINUITY_CHECK_SYSTEM_PROMPT = "You are a conversation continuity detector. Return ONLY 'true' or 'false'."
CONTINUITY_CHECK_USER_PROMPT = ("Determine if these two conversation pages are continuous (true continuation without topic shift).\n"
                                "Return ONLY \"true\" or \"false\".\n\n"
                                "Previous Page:\nUser: {prev_user}\nAssistant: {prev_agent}\n\n"
                                "Current Page:\nUser: {curr_user}\nAssistant: {curr_agent}\n\n"
                                "Continuous?")

# 生成元信息提示词
# 翻译：系统：你是一个对话元摘要更新器。你的任务是：
# 1. 保留前一元摘要中的相关上下文。
# 2. 整合当前对话中的新信息。
# 3. 仅输出更新后的摘要（无解释）。
# 用户：通过整合新对话并保持连贯性来更新对话元摘要。指南：从前一元摘要开始（如果存在），根据新对话添加/更新信息，保持简练（最多1-2句），保持上下文连贯。
# 前一元摘要：{last_meta} 新对话：{new_dialogue} 更新后的元摘要：
# Prompt for generating meta info (from dynamic_update.py, _generate_meta_info)
META_INFO_SYSTEM_PROMPT = ("""You are a conversation meta-summary updater. Your task is to:
1. Preserve relevant context from previous meta-summary
2. Integrate new information from current dialogue
3. Output ONLY the updated summary (no explanations)""" )
META_INFO_USER_PROMPT = ("""Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.
        
    Guidelines:
    1. Start from the previous meta-summary (if exists)
    2. Add/update information based on the new dialogue
    3. Keep it concise (1-2 sentences max)
    4. Maintain context coherence

    Previous Meta-summary: {last_meta}
    New Dialogue:
    {new_dialogue}

    Updated Meta-summary:""")
