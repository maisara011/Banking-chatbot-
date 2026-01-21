# # llm/llm_handler.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

class LLMHandler:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b", # openai/gpt-oss-120b, llama-3.1-8b-instant
            temperature=0.3,
            api_key=api_key
        )

    def generate(self, user_query: str) -> str:
        response = self.llm.invoke([
            HumanMessage(
                content=f"""
Answer the following user question clearly and factually.
If applicable, mention that this is a general informational answer.

Question:
{user_query}
"""
            )
        ])
        return response.content
