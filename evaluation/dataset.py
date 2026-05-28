"""
scripts/evaluation/dataset.py

Monta o EvaluationDataset do RAGAs a partir do pipeline RAG existente.

Fluxo:
  1. Lê o arquivo de perguntas + ground truth (JSON ou CSV)
  2. Para cada pergunta, chama o pipeline e captura contexts + answer
  3. Retorna um EvaluationDataset pronto para ragas.evaluate()

Formato esperado do arquivo de perguntas (JSON):
  [
    {
      "question": "O que é X?",
      "ground_truth": "X é ..."   ← opcional, necessário para métricas como Context Recall e Answer Accuracy
    },
    ...
  ]
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

from ragas import EvaluationDataset, SingleTurnSample

from app.core.retrieval.retriever import retriever
from app.core.generation.generator import generator


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_questions(path: str | Path) -> list[dict[str, str]]:
    """
    Carrega perguntas de um arquivo .json ou .csv.

    JSON esperado: lista de objetos com "question" e opcionalmente "ground_truth".
    CSV esperado: colunas "question" e opcionalmente "ground_truth".
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    if p.suffix == ".json":
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON deve ser uma lista de objetos.")
        return data

    if p.suffix == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    raise ValueError(f"Formato não suportado: {p.suffix}. Use .json ou .csv")


def _run_single(question: str) -> dict[str, Any]:
    """
    Executa retrieval + generation para uma pergunta e retorna
    os campos que o RAGAs precisa.
    """
    sources = retriever.search(question)

    if not sources:
        return {
            "question": question,
            "contexts": [],
            "response": "I could not find relevant information to answer your question.",
        }

    answer = generator.generate(question=question, sources=sources)
    contexts = [s.content for s in sources]

    return {
        "question": question,
        "contexts": contexts,
        "response": answer,
    }


# ── API pública ────────────────────────────────────────────────────────────────

def build_evaluation_dataset(
    questions_path: str | Path,
    *,
    verbose: bool = True,
) -> EvaluationDataset:
    """
    Constrói o EvaluationDataset do RAGAs a partir de um arquivo de perguntas.

    Args:
        questions_path: caminho para o arquivo .json ou .csv com as perguntas.
        verbose: imprime progresso no terminal.

    Returns:
        EvaluationDataset pronto para ragas.evaluate().
    """
    questions = _load_questions(questions_path)
    samples: list[SingleTurnSample] = []

    for i, item in enumerate(questions, start=1):
        question = item.get("question", "").strip()
        ground_truth = item.get("ground_truth", "").strip() or None

        if not question:
            print(f"[WARN] Item {i} sem campo 'question', ignorado.")
            continue

        if verbose:
            print(f"[{i}/{len(questions)}] Processando: {question[:60]}...")

        result = _run_single(question)

        sample = SingleTurnSample(
            user_input=result["question"],
            retrieved_contexts=result["contexts"],
            response=result["response"],
            reference=ground_truth,  # None se não fornecido
        )
        samples.append(sample)

    if not samples:
        raise ValueError("Nenhuma pergunta válida foi processada.")

    if verbose:
        print(f"\n✓ Dataset montado com {len(samples)} amostras.")

    return EvaluationDataset(samples=samples)