# app/core/agent/planner.py

import json

from google.genai import types

from config import client, settings

PLANNER_PROMPT = """
You are a routing agent.

Decide whether the question requires retrieving documents
from a vector database.

Return ONLY valid JSON.

Examples:

Question:
What is FastAPI?

{
  "action": "retrieve"
}

Question:
What is 2 + 2?

{
  "action": "direct"
}

Question:
Hello

{
  "action": "direct"
}
"""


class Planner:
    def decide(self, question: str) -> str:

        prompt = f"""
{PLANNER_PROMPT}

Question:
{question}
"""

        response = client.models.generate_content(
            model=settings.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )

        try:
            result = json.loads(response.text)

            action = result.get("action", "retrieve")

            if action not in ["retrieve", "direct"]:
                return "retrieve"

            return action

        except Exception:
            return "retrieve"


planner = Planner()
