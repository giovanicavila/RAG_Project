# app/core/agent/state.py
# No external dependencies — this file was already provider-agnostic.
# Keeping it unchanged, only adding type hints for clarity.

from dataclasses import dataclass, field
from app.models.schemas import SourceDocument


@dataclass
class AgentState:
    original_question: str
    current_question: str
    context: list[SourceDocument] = field(default_factory=list)
    attempts: int = 0
    action: str = "retrieve"
    # action tracks the planner decision for the current turn:
    # "retrieve" → run retrieval pipeline
    # "direct"   → answer directly without retrieval