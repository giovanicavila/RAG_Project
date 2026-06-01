# app/core/agent/state.py

from dataclasses import dataclass, field

@dataclass
class AgentState:
    original_question: str
    current_question: str
    context: list = field(default_factory=list)
    attempts: int = 0