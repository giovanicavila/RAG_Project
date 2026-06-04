# RAG Project — Architecture & Design Document

> **Purpose:** Explain how the system is organized, why design decisions were made, and how to extend it.

---

## Table of Contents
1. [Overview & Philosophy](#1-overview--philosophy)
2. [Technology Stack](#2-technology-stack)
3. [Directory Structure](#3-directory-structure)
4. [Core Architectural Patterns](#4-core-architectural-patterns)
5. [Layer-by-Layer Breakdown](#5-layer-by-layer-breakdown)
6. [Data Flow](#6-data-flow)
7. [The Provider Pattern](#7-the-provider-pattern)
8. [Agentic RAG Pipeline](#8-agentic-rag-pipeline)
9. [Configuration System](#9-configuration-system)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Extension Guide](#11-extension-guide)

---

## 1. Overview & Philosophy

This is a **Retrieval-Augmented Generation (RAG) API** built with FastAPI. It answers user questions by retrieving relevant document chunks from a local ChromaDB vector store and generating grounded responses via an LLM.

### Core Philosophy

- **Model-agnostic:** Never tie the codebase to a single LLM or embedding provider. All external model calls go through abstract interfaces.
- **Provider Pattern:** Swapping from OpenRouter to Ollama or from local embeddings to OpenAI should require zero code changes — only environment variable updates.
- **Zero business logic in HTTP layer:** `routes.py` only validates and delegates. All intelligence lives in `core/`.
- **Agentic self-correction:** The system doesn't just retrieve once. It *plans*, *retrieves*, *grades*, and *rewrites* until it finds relevant context.
- **Configuration via environment:** No hardcoded API keys, model names, or URLs anywhere.

---

## 2. Technology Stack

| Component       | Technology                             |
|-----------------|----------------------------------------|
| Web Framework   | FastAPI + Pydantic v2                  |
| Vector Store    | ChromaDB (local, persistent)           |
| Embeddings      | sentence-transformers (default) or OpenAI-compatible APIs |
| LLM Generation  | OpenAI-compatible APIs (OpenRouter, Ollama, OpenAI) |
| Chunking        | Chonkie (SemanticChunker)              |
| Evaluation      | RAGAS                                  |
| Orchestration   | Custom agent loop (not LangChain)      |

---

## 3. Directory Structure

```
RAG_PROJECT/
├── app/
│   ├── api/
│   │   └── routes.py                      ← HTTP layer (validates, delegates)
│   ├── core/
│   │   ├── agent/
│   │   │   ├── state.py                   ← AgentState dataclass
│   │   │   ├── planner.py                 ← Decides: retrieve or direct answer
│   │   │   ├── grader.py                  ← Grades if context is relevant
│   │   │   ├── rewriter.py                ← Rewrites query for better retrieval
│   │   │   └── tools.py                   ← Retrieval tool wrapper
│   │   ├── chunker/
│   │   │   └── chunker.py                 ← Semantic text splitter (Chonkie)
│   │   ├── embeddings/
│   │   │   ├── base.py                    ← BaseEmbedder (ABC)
│   │   │   ├── sentence_transformers_embedder.py  ← Local, no key
│   │   │   ├── openai_embedder.py        ← OpenAI-compatible endpoint
│   │   │   └── factory.py                 ← get_embedder() → singleton
│   │   ├── generation/
│   │   │   ├── base.py                    ← BaseLLM (ABC)
│   │   │   ├── openai_compatible_llm.py  ← One class for all OAI-compatible providers
│   │   │   └── factory.py                 ← get_llm() / get_agent_llm() → singletons
│   │   ├── prompts/
│   │   │   └── templates.py               ← RAG_SYSTEM_PROMPT + build_context()
│   │   ├── retrieval/
│   │   │   └── retriever.py               ← ChromaDB wrapper → singleton
│   │   └── pipeline.py                    ← Orchestrates the full RAG + agent flow
│   └── models/
│       └── schemas.py                     ← Pydantic request/response models
├── evaluation/                             ← RAG quality metrics (RAGAS)
│   ├── cli.py
│   ├── runner.py
│   ├── config.py
│   ├── serializers.py
│   ├── datasets/
│   │   ├── loader.py
│   │   └── builder.py
│   ├── metrics/
│   │   ├── groups.py
│   │   ├── llm.py
│   │   └── registry.py
│   └── data/
│       └── questions.json
├── scripts/
│   └── ingestion/
│       └── ingest.py                      ← Indexes documents into ChromaDB
├── config.py                              ← Pydantic Settings + .env
├── main.py                                ← FastAPI app entrypoint
├── requirements.txt
└── ARCHITECTURE.md                        ← This file
```

---

## 4. Core Architectural Patterns

### 4.1 Abstract Base Class (ABC) + Factory

Both LLM and Embedding providers implement a base interface. The factory reads config and returns the correct instance. The rest of the app interacts only with the base class.

```
BaseLLM.generate(prompt) ──► OpenAICompatibleLLM
                     ──► (future: AnthropicLLM, LocalTransformersLLM)

BaseEmbedder.embed_text(text) ──► SentenceTransformersEmbedder
                          ──► OpenAIEmbedder
```

### 4.2 Singleton for Heavy Clients

Expensive objects (embedding model, LLM client, ChromaDB collection) are instantiated **once** at import time and reused everywhere via module-level variables:

```python
# Import these, never instantiate directly
from app.core.embeddings.factory import embedder
from app.core.generation.factory import llm, agent_llm
from app.core.retrieval.retriever import retriever
```

Benefits:
- Avoids cold-start latency on every request
- Prevents connection pool exhaustion
- HuggingFace models are cached in `~/.cache/` after first download

### 4.3 Dependency Direction

```
main.py
  │  imports → routes.py
  │            │ calls → pipeline.py
  │            │       │ uses → planner / grader / rewriter / retrieval_tool
  │            │       │   │     └── agent_llm (from factory)
  │            │       │   │
  │            │       │   └── retriever (from factory)
  │            │       │       └── embedder (from factory)
  │            │       │
  │            │       └── llm (from factory)
  │            │           └── generates final answer
  │            │
  │            └── schemas.py (shared Pydantic models)
```

**Rule:** Lower layers never import upper layers. `pipeline.py` never imports `routes.py`.

---

## 5. Layer-by-Layer Breakdown

### 5.1 API Layer — `app/api/routes.py`

**Responsibility:** Accept HTTP requests, validate bodies, delegate to core, return typed responses.

```python
@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    return run_rag_pipeline(question=request.question, top_k=request.top_k)

@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(request: IngestRequest) -> IngestResponse:
    # 1. chunk text
    # 2. create metadata
    # 3. generate UUID ids
    # 4. call ingestion.add_documents()
    # 5. return summary
```

**Consciously Simple:** The endpoint uses `semantic chunking`, tracks token counts in metadata, and assigns unique IDs with `uuid4()`. No orchestration decisions here.

### 5.2 Core Layer — `app/core/`

This is where all business logic lives.

#### 5.2.1 Pipeline (`pipeline.py`)

The main entry point. Implements an **agentic loop** with up to 3 retrieval attempts:

```
1. Planner decides: "retrieve" or "direct answer"
2. If "direct" → ask LLM, done
3. For up to 3 attempts:
   a. Retrieve chunks from ChromaDB
   b. Grade if context is relevant to question
   c. If relevant → build prompt, generate answer, done
   d. If not relevant → rewriter rewrites the query, retry
4. Fallback: if nothing found after 3 attempts → direct answer
```

#### 5.2.2 Agents (`agent/`)

| Agent       | Purpose                                                                   |
|-------------|---------------------------------------------------------------------------|
| **Planner** | Routes questions. "direct" for math/facts the LLM already knows, "retrieve" for domain-specific knowledge. |
| **Grader**  | Binary judge — is the retrieved context sufficient to answer the question? |
| **Rewriter**| Transforms vague questions into more specific, retrieval-friendly queries. |
| **Tools**   | Thin wrapper around `retriever.search()` decoupling the agent from storage. |
| **State**   | Dataclass tracking original question, current query, attempts, and action across turns. |

Each agent uses `agent_llm` — possibly a different (often cheaper/faster) model than the main generation LLM.

#### 5.2.3 Generation (`generation/`)

- **`base.py`** — `BaseLLM` with abstract `generate(prompt) -> str`
- **`openai_compatible_llm.py`** — One class covers all OpenAI-Chat-Completions-compatible APIs
- **`factory.py`** — Returns `llm` (main) and `agent_llm` depending on config

#### 5.2.4 Embeddings (`embeddings/`)

- **`base.py`** — `BaseEmbedder` with `embed_text()` and `embed_batch()`
- **`sentence_transformers_embedder.py`** — Runs locally, downloads on first use, normalizes vectors for cosine similarity
- **`openai_embedder.py`** — Calls OpenAI-compatible `/embeddings` endpoint
- **`factory.py`** — Returns the configured singleton

#### 5.2.5 Retrieval (`retrieval/retriever.py`)

Wraps ChromaDB. Given a query string:
1. Embeds via `embedder.embed_text(query)`
2. Runs `collection.query(...)` with cosine distance
3. Converts distances to similarity scores (`score = 1 - distance`)
4. Returns `list[SourceDocument]`

#### 5.2.6 Chunking (`chunker/chunker.py`)

Uses `Chonkie.SemanticChunker` which splits by semantic meaning using the embedding model itself. This is superior to fixed-token chunking for question-answering because boundaries respect sentence/paragraph coherence.

#### 5.2.7 Prompts (`prompts/templates.py`)

Contains the system prompt with explicit instructions on when to use retrieved context vs parametric knowledge. Includes examples of how to prefix answers when context is insufficient. `build_context()` formats chunks into a numbered block.

### 5.3 Models — `app/models/schemas.py`

Pure Pydantic models shared across the app. No business logic here.

| Class             | Role                                            |
|-------------------|-------------------------------------------------|
| `QueryRequest`    | Validates `question` (3–1000 chars) + optional `top_k` (1–20) |
| `QueryResponse`   | Returns `answer`, `sources[]`, and `question` (for debugging) |
| `SourceDocument`  | One retrieved chunk with `id`, `content`, `source`, `score` |
| `IngestRequest`   | Validates `text` (≥10 chars) + `source_name` |
| `IngestResponse`  | Returns `message`, `chunks_created`, `source` |

### 5.4 Evaluation — `evaluation/`

A standalone module for quality assessment using **RAGAS**.

| File | Purpose |
|------|---------|
| `cli.py` | Argument parser for `--questions`, `--mode`, `--output` |
| `runner.py` | Loads questions → builds dataset (query + retrieve + generate) → evaluates → saves |
| `datasets/builder.py` | Runs the live RAG pipeline for each question to build an `EvaluationDataset` |
| `datasets/loader.py` | Loads `.json` or `.csv` question files |
| `metrics/registry.py` | Binds RAGAS metrics to the evaluation LLM |
| `metrics/groups.py` | Defines evaluation modes: `quick`, `retrieval`, `generation`, `full` |
| `serializers.py` | Saves timestamped JSON results to `evaluation/results/` |

> **Note:** The evaluation module currently uses `langchain_google_genai` for its judge LLM — this is a known dependency inherited from RAGAS integration and is isolated to the evaluation layer only.

### 5.5 Ingestion — `scripts/ingestion/ingest.py`

A standalone script to index documents. Can be run independently or triggered via the `/ingest` API endpoint.

---

## 6. Data Flow

### 6.1 Query Flow (Runtime)

```
Client Request
      │
      ▼
┌──────────────┐
│ /query       │
│ FastAPI      │
│ + Pydantic   │      QueryRequest
└──────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ run_rag_pipeline(question, top_k)                   │
│                                                     │
│  ┌──────────┐                                       │
│  │ Planner  │  → "retrieve" or "direct"              │
│  │ (agent)  │                                       │
│  └──────────┘                                       │
│         │                                           │
│         ▼ direct?                                   │
│      ┌──────────────┐                               │
│      │  LLM.generate│ → parametric answer, DONE    │
│      └──────────────┘                               │
│         │                                           │
│         ▼ retrieve?                                  │
│  ┌─────────────────────────────────────────────┐   │
│  │ Loop: up to 3 attempts                      │   │
│  │                                             │   │
│  │  ┌──────────────┐  ┌─────────┐  ┌─────────┐ │   │
│  │  │ retrieve     │─▶│ grade   │ │ rewrite │ │   │
│  │  │ (ChromaDB)   │  │ relevant?│ │ query   │ │   │
│  │  └──────────────┘  └────┬────┘  └──┬──────┘ │   │
│  │                         │ yes      │ no     │   │
│  │                         ▼          └───────▶│   │
│  │                  break loop                 │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼ no sources after 3 tries   │
│                  ┌──────────────┐                   │
│                  │ fallback to  │                   │
│                  │ parametric   │                   │
│                  │ answer       │                   │
│                  └──────────────┘                   │
│                         ▼                           │
│                  ┌──────────────┐                   │
│                  │ LLM.generate │ ← with context   │
│                  │ (RAG prompt) │                   │
│                  └──────────────┘                   │
└─────────────────────────────────────────────────────┘
      │
      ▼
QueryResponse(answer, sources[], question)
```

### 6.2 Ingestion Flow

```
Client Request / File System
      │
      ▼
IngestRequest(text, source_name)
      │
      ▼
SemanticChunker.split(text)  →  list of chunks + token_count
      │
      ▼
Assign metadata: {source, chunk_index, token_count}
      │
      ▼
embedder.embed_batch(chunks)
      │
      ▼
ChromaDB.collection.upsert(ids, embeddings, documents, metadatas)
      │
      ▼
IngestResponse(message, chunks_created, source)
```

---

## 7. The Provider Pattern

This is the **most important pattern** in the project. It makes the system immune to provider churn.

### 7.1 How It Works

```
App Code ──► BaseLLM.generate()
                │
                └──► Factory decides at startup:
                       if provider == "ollama":     → OpenAICompatibleLLM(base_url=localhost)
                       if provider == "openrouter": → OpenAICompatibleLLM(base_url=openrouter.ai)
                       if provider == "openai":     → OpenAICompatibleLLM(base_url=default)
```

### 7.2 Benefits

- **Zero code changes** to swap providers — only `.env` updates
- **Testability** — mock `BaseLLM` or `BaseEmbedder` in unit tests
- **Future-proofing** — Anthropic, Gemini, or local GGUF models can be added by writing one new class
- **Single source of truth** — provider URLs live in one place (`factory.py`)

### 7.3 Adding a New Provider

1. Implement the ABC (`BaseLLM` or `BaseEmbedder`) in a new file
2. Add the provider literal to `config.py`
3. Add the branch in the corresponding `factory.py`
4. Set the new provider name in `.env`

---

## 8. Agentic RAG Pipeline

### 8.1 Loop Detail

The pipeline (`pipeline.py`) runs an adaptive agent loop with up to 3 retrieval attempts:

```
1. Planner decides: "retrieve" or "direct"
2. If "direct" → LLM generates from parametric memory, done
3. For attempt in 1..3:
   a. Retrieve top_k chunks from ChromaDB via retrieval_tool
   b. If no chunks found → rewrite query, retry
   c. Grader evaluates context relevance
   d. If relevant → break loop
   e. If not → Rewriter reformulates query → retry
4. If no sources after all attempts → fallback to parametric answer
5. Build RAG prompt with context → LLM generates grounded answer
```

### 8.2 Agent State

The `AgentState` dataclass tracks state across turns:

| Field | Purpose |
|-------|---------|
| `original_question` | Unchanged user input |
| `current_question` | May be rewritten by rewriter |
| `context` | Retrieved chunks (`list[SourceDocument]`) |
| `attempts` | Retry counter |
| `action` | Planner decision: `"retrieve"` or `"direct"` |

### 8.3 Agent Prompts and Outputs

| Agent | Model | Prompt Style | Output |
|-------|-------|-------------|--------|
| **Planner** | `agent_llm` | JSON instruction with examples | `{"action": "retrieve" \| "direct"}` |
| **Grader** | `agent_llm` | JSON instruction | `{"relevant": true \| false}` |
| **Rewriter** | `agent_llm` | Free-text instruction | Rewritten query string |

### 8.4 Why Agentic?

- **Single-shot RAG** fails when retrieved chunks are irrelevant. The grader catches this.
- **Query rewriting** reformulates vague questions (e.g., "how fast?" → "what is the top speed of the X-200 engine?"), improving retrieval recall.
- **Fail-safe:** after 3 failed attempts, the system returns a parametric answer rather than an empty response.

---

## 9. Configuration System

### 9.1 `config.py`

Uses **Pydantic Settings v2** for type-safe, validated configuration loaded from `.env`.

| Group | Settings |
|-------|----------|
| LLM (generation) | `llm_provider`, `llm_model`, `llm_api_key`, `llm_base_url` |
| LLM (agent) | `agent_llm_provider`, `agent_llm_model`, `agent_llm_api_key`, `agent_llm_base_url` |
| Embedding | `embedding_provider`, `embedding_model`, `embedding_dimension`, `embedding_api_key` |
| Vector Store | `vector_store_provider`, `vector_store_persist_directory`, `vector_store_collection_name`, `vector_store_similarity_metric` |
| RAG | `top_k`, `max_tokens`, `temperature` |

### 9.2 Auto-fallback Logic

- If `agent_llm_api_key` is empty → falls back to `llm_api_key`
- If `agent_llm_base_url` is empty → falls back to `llm_base_url`
- Ensures agent LLM can share the same credentials as the main LLM without redundant config

### 9.3 Validation

Raises `ValueError` at startup for:
- Missing `LLM_API_KEY` when provider is `openrouter` or `openai`
- Missing `AGENT_LLM_API_KEY` when agent provider is `openrouter` or `openai`
- Missing `EMBEDDING_API_KEY` when embedding provider is `openai` or `openrouter`
- Unsupported provider strings

### 9.4 `.env` File

All config is loaded from `.env` at process startup. See `.env.example` for all available options.

---

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No LangChain** | LangChain adds abstraction overhead and debugging complexity. Custom agent loop is ~300 lines vs 1000+ with LCEL. |
| **Semantic chunking** | Fixed-token chunking splits sentences mid-thought. Semantic chunking (Chonkie) respects sentence/paragraph boundaries using embedding similarity shifts. |
| **Cosine similarity** | ChromaDB default. Embeddings are L2-normalized, converting cosine distance (0..2) to similarity score (`1 - distance`). |
| **Separate agent LLM** | Planner/grader/rewriter calls are frequent but simple. A cheaper model reduces cost without sacrificing quality. |
| **Singleton factories** | Embedding models are ~100MB+ to load; ChromaDB clients hold connection pools. Per-request instantiation would be wasteful. |
| **Pydantic v2** | Input validation, serialization, and OpenAPI docs from a single schema definition. |
| **Synchronous core** | ChromaDB and sentence-transformers are synchronous. Wrapping in async adds complexity without benefit for a local-first API. |
| **Fallback on failure** | If retrieval yields nothing after 3 attempts, the system falls back to parametric knowledge rather than returning empty-handed. |

---

## 11. Extension Guide

### 11.1 Adding a New LLM Provider

1. Create `app/core/generation/anthropic_llm.py` implementing `BaseLLM`
2. Add the provider to the `llm_provider` / `agent_llm_provider` Literal in `config.py`
3. Add the default `base_url` in `generation/factory.py._BASE_URLS`
4. Add API key validation in `config.py`'s `@model_validator`
5. Set `LLM_PROVIDER=anthropic` in `.env` — no other code changes needed

### 11.2 Adding a New Embedding Provider

1. Create `app/core/embeddings/cohere_embedder.py` implementing `BaseEmbedder`
2. Add provider to `embedding_provider` Literal in `config.py`
3. Add branch in `embeddings/factory.py`
4. Set `EMBEDDING_PROVIDER=cohere` in `.env`

### 11.3 Adding a New Agent

1. Create `app/core/agent/your_agent.py`
2. Import `agent_llm` from `app.core.generation.factory`
3. Wire it into `pipeline.py`'s loop
4. Optionally add fields to `AgentState`

### 11.4 Adding a New API Route

1. Define request/response models in `app/models/schemas.py`
2. Add route handler in `app/api/routes.py`
3. Delegate to a core function — keep the route thin

### 11.5 Adding a New Vector Store

1. Create a new retriever (e.g., `app/core/retrieval/pinecone_retriever.py`)
2. Implement `search()` returning `list[SourceDocument]`
3. Update `vector_store_provider` Literal in `config.py`
4. Swap the retriever singleton used by the rest of the app