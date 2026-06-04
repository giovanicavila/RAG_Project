from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the generated text."""
        ...
