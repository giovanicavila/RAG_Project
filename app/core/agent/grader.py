# app/core/agent/grader.py
import json

from app.core.generation.factory import agent_llm as llm


class ContextGrader:
    def grade(self, question: str, context: str) -> bool:
        """
        Returns True if the context contains enough information
        to answer the question, False otherwise.
        """
        prompt = f"""Question:
{question}

Context:
{context}

Can this context answer the question?

Return JSON only — no explanation, no markdown, no extra text:

{{"relevant": true}}

or

{{"relevant": false}}
"""
        response = llm.generate(prompt)

        # Strip markdown fences if the model wraps the JSON (e.g. ```json ... ```)
        cleaned = (
            response.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        try:
            result = json.loads(cleaned)
            return bool(result.get("relevant", False))
        except (json.JSONDecodeError, KeyError):
            # Fail safe — treat unparseable response as not relevant.
            return False


grader = ContextGrader()
