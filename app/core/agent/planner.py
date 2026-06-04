# app/core/agent/planner.py
import json
from app.core.generation.factory import agent_llm as llm

PLANNER_PROMPT = """You are a routing agent.

Decide whether the question requires retrieving documents
from a vector database.

Return ONLY valid JSON — no explanation, no markdown, no extra text.

Examples:

Question: What is FastAPI?
{"action": "retrieve"}

Question: What is 2 + 2?
{"action": "direct"}

Question: Hello
{"action": "direct"}
"""


class Planner:
    def decide(self, question: str) -> str:
        """
        Returns "retrieve" if the question needs document lookup,
        or "direct" if the LLM can answer without context.
        Defaults to "retrieve" on any parsing failure.
        """
        prompt = f"{PLANNER_PROMPT}\nQuestion: {question}"
        response = llm.generate(prompt)

        cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            result = json.loads(cleaned)
            action = result.get("action", "retrieve")
            if action not in ("retrieve", "direct"):
                return "retrieve"
            return action
        except (json.JSONDecodeError, KeyError):
            # Fail safe — default to retrieval when uncertain.
            return "retrieve"


planner = Planner()