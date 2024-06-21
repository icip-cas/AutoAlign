META_JUDGE_EN = """\
Review the user's question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion: 
- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content. 
- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer. 
- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results. 
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus. 
- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer. 

User: {instruction} 

<response>{response}</response> 

After examining the user's instruction and the response: 

- Briefly justify your total score, up to 100 words. 
- Conclude with the score using the format: “Score: <total points>”

Remember to assess from the AI Assistant perspective and give a score between 0 and 5. To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria.
"""

META_JUDGE_ZH = """
根据下面描述的累积5分制,审查用户的问题及对应的回答。根据满足的每个标准累积相应的分数:
- 如果回答与用户的询问相关并提供了一些相关信息,即使不完整或包含一些无关内容,加1分。
- 如果回答涵盖了用户问题的大部分内容,但并没有完全解决疑问或直接作答,再加1分。
- 如果回答以一种有用的方式回答了用户问题的基本要素,无论这看起来是否像是由人工智能助理撰写的,或者包含一些博客或搜索结果中典型的元素,加1分。
- 如果回答明确由人工智能助手的角度出发,直接全面地回答了用户的问题,组织良好有助于理解,即使在清晰度、简洁性或重点方面还有一些改进空间,加1分。
- 如果回答由人工智能助手量身定制并完美回答了用户的问题,没有多余信息,体现了专业知识,提供了高质量、富有洞识力的精彩答复,满分5分。

用户问题：{instruction}
<回答>{response}</回答> 

在检查了用户的指令和回答后:

- 简要说明你的总分数理由(最多100个字)
- 以"分数: <总分数>"的格式写出总分数

请记住,从人工智能助理的角度进行评估,给出0-5之间的打分。我们将根据所概述的标准系统地累加分数,以符合这种累积评分模式。
"""

META_JUDGE_EN_WEB = """\
Review the user's question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion: 
- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content. 
- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer. 
- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results. 
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus. 
- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer. 

User: {instruction} 

<response>{response}</response> 

After examining the user's instruction and the response: 

- Briefly justify your total score, up to 100 words. 
- Conclude with the score using the format: “Score: <total points>”

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria.
"""

META_JUDGE_ZH_WEB = """
根据下面描述的累积5分制,审查用户的问题及对应的回答。根据满足的每个标准累积相应的分数:
- 如果回答与用户的询问相关并提供了一些相关信息,即使不完整或包含一些无关内容,加1分。
- 如果回答涵盖了用户问题的大部分内容,但并没有完全解决疑问或直接作答,再加1分。
- 如果回答以一种有用的方式回答了用户问题的基本要素,无论这看起来是否像是由人工智能助理撰写的,或者包含一些博客或搜索结果中典型的元素,加1分。
- 如果回答明确由人工智能助手的角度出发,直接全面地回答了用户的问题,组织良好有助于理解,即使在清晰度、简洁性或重点方面还有一些改进空间,加1分。
- 如果回答由人工智能助手量身定制并完美回答了用户的问题,没有多余信息,体现了专业知识,提供了高质量、富有洞识力的精彩答复,满分5分。

用户问题：{instruction}
<回答>{response}</回答> 

在检查了用户的指令和回答后:

- 简要说明你的总分数理由(最多100个字)
- 以"分数: <总分数>"的格式写出总分数

请记住,从人工智能助理的角度进行评估,必要时利用网络搜索知识。我们将根据所概述的标准系统地累加分数,以符合这种累积评分模式。
"""
