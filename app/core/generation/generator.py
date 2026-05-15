from google.genai import types
from app.core.prompts.templates import RAG_SYSTEM_PROMPT, build_context
from app.models.schemas import SourceDocument
from config import settings, client

class Generator:
    def __init__(self):
        self.model_name = settings.model_name

    def generate(self, question: str, sources: list[SourceDocument]) -> str:
        context = build_context(sources)
        full_prompt = f"{RAG_SYSTEM_PROMPT.format(context=context)}\n\nQuestion: {question}"

        response = client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=settings.max_tokens,
                temperature=settings.temperature, 
            ),
        )
        return response.text

generator = Generator()