# app/core/agent/rewriter.py
from app.core.generation.factory import llm


class QueryRewriter:
    def rewrite(self, question: str) -> str:
        """
        Rewrites the user question to improve semantic retrieval quality.
        Returns the rewritten query as a plain string.
        """
        prompt = f"""Rewrite the question below to improve semantic retrieval.
Make it more specific and descriptive.
Return only the rewritten query — no explanation, no punctuation changes.

Question:
{question}"""

        return llm.generate(prompt).strip()


rewriter = QueryRewriter()