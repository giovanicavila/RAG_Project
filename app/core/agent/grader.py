import json

from google.genai import types

from config import client, settings


class ContextGrader:
    def grade(self, question: str, context: str) -> bool:

        prompt = f"""
Question:
{question}

Context:
{context}

Can this context answer the question?

Return JSON only:

{{"relevant": true}}

or

{{"relevant": false}}
"""

        response = client.models.generate_content(
            model=settings.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )

        try:
            result = json.loads(response.text)
            return result["relevant"]
        except:
            return False


grader = ContextGrader()
