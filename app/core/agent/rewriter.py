from google.genai import types

from config import client, settings


class QueryRewriter:
    def rewrite(self, question: str) -> str:

        prompt = f"""
Rewrite the question to improve semantic retrieval.

Question:
{question}

Return only the rewritten query.
"""

        response = client.models.generate_content(
            model=settings.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )

        return response.text.strip()


rewriter = QueryRewriter()
