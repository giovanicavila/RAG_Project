import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.core.prompts.templates import RAG_SYSTEM_PROMPT, build_context
from app.models.schemas import SourceDocument
from config import settings

genai.configure(api_key=settings.gemini_api_key)


class Generator:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=settings.model_name,
            # "gemini-2.0-flash" — free tier, 1M token context window!
        )
        self.generation_config = GenerationConfig(
            max_output_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )

    def generate(self, question: str, sources: list[SourceDocument]) -> str:
        context = build_context(sources)
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

        # In the Gemini SDK, we concatenate the system prompt and question
        # into a single prompt string.
        # For multi-turn conversations (chat history), use model.start_chat(history=[...]).
        full_prompt = f"{system_prompt}\n\nQuestion: {question}"

        response = self.model.generate_content(
            full_prompt,
            generation_config=self.generation_config,
        )
        return response.text  # .text → the generated response as a plain string.


generator = Generator()
