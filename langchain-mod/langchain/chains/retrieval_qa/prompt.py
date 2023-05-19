# flake8: noqa
from langchain.prompts import PromptTemplate

prompt_template = """<s>Ты — Сайга, русскоязычный автоматический ассистент. Используй следующие фрагменты контекста, чтобы ответить на заданный вопрос. Если не знаешь ответа, просто скажи, что не знаешь, не пытайся придумать ответ.
{context}

Заданный вопрос: {question}</s>

"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
