from langchain import PromptTemplate

"""
这个模块（或代码片段）的目的是创建一个PromptTemplate实例，用于生成基于给定上下文和问题的提示。
"""

qa_template = """
    已知信息：
    {context} 

    根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 
    问题是：{question}
    """

prompt_qa = PromptTemplate(template=qa_template, input_variables=["context", "question"])