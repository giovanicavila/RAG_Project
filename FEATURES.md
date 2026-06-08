# New Features — RAG Project

## 1. Provider Pattern

We introduced an **abstraction layer** that decouples the entire system from any specific LLM or embedding provider. The application never imports an SDK directly — it always talks to a base interface.

### How it works

```
App Code ──► BaseLLM.generate(prompt)
                │
                └──► Factory (reads .env)
                       ├── "ollama"     → OpenAICompatibleLLM(base_url=localhost:11434)
                       ├── "openrouter" → OpenAICompatibleLLM(base_url=openrouter.ai)
                       └── "openai"     → OpenAICompatibleLLM(base_url=api.openai.com)
```

```
App Code ──► BaseEmbedder.embed_text(text)
                │
                └──► Factory (reads .env)
                       ├── "sentence_transformers" → SentenceTransformersEmbedder (local)
                       └── "openai"                → OpenAIEmbedder (API-based)
```

### Key design decisions

- **Two separate LLM slots:** `llm` (for final answer generation) and `agent_llm` (for agentic tools — planner, grader, rewriter). Each has independent `provider`, `model`, `api_key`, and `base_url` config.
- **Single implementation class:** `OpenAICompatibleLLM` handles **all** providers because Ollama, OpenRouter, OpenAI, vLLM, LM Studio, Groq etc. all speak the same OpenAI Chat Completions format. No per-provider class needed.
- **Zero code changes to swap:** Change `LLM_PROVIDER` and `LLM_BASE_URL` in `.env`, restart — everything else stays the same.
- **Local-first embeddings:** `sentence-transformers` runs 100% offline with no API key, making development and testing free.

### Where the code lives

| File | Purpose |
|------|---------|
| `app/core/generation/base.py` | `BaseLLM` ABC |
| `app/core/generation/openai_compatible_llm.py` | Concrete implementation |
| `app/core/generation/factory.py` | `get_llm()` / `get_agent_llm()` singletons |
| `app/core/embeddings/base.py` | `BaseEmbedder` ABC |
| `app/core/embeddings/sentence_transformers_embedder.py` | Local embeddings |
| `app/core/embeddings/openai_embedder.py` | API embeddings |
| `app/core/embeddings/factory.py` | `get_embedder()` singleton |
| `config.py` | All `_provider`, `_model`, `_api_key`, `_base_url` settings |

---

## 2. Agentic RAG Pipeline

Instead of a single-shot retrieve-then-generate flow, the system now runs an **adaptive agent loop** with up to 3 retrieval attempts. Three specialized agents collaborate to improve retrieval quality:

### The three agents

| Agent | Role |
|-------|------|
| **Planner** | Routes the question. If it's a general/factual question (math, definitions, greetings), returns `"direct"` — answer from parametric memory. If it requires domain-specific knowledge, returns `"retrieve"`. |
| **Grader** | Binary relevance judge. After chunks are retrieved, evaluates whether the context actually answers the question. Returns `true` / `false`. |
| **Rewriter** | Query reformulator. Transforms vague or poorly-phrased questions into more specific, retrieval-friendly queries. |

### How the loop works

```
1. Planner decides: "retrieve" or "direct"
2. If "direct" → LLM generates answer from memory, done
3. For up to 3 attempts:
   a. Retrieve chunks from ChromaDB
   b. If no chunks found → Rewriter reformulates → retry
   c. Grader checks if context is relevant
   d. If relevant → build RAG prompt with context → done
   e. If not → Rewriter reformulates query → retry
4. Fallback: no relevant context after 3 attempts → direct answer
```

### Why this matters

- **Single-shot RAG** fails silently when retrieved chunks are irrelevant — the LLM either hallucinates or produces nonsense. The grader catches this and triggers a retry.
- **Query rewriting** improves recall on vague questions (e.g. `"how fast?"` → `"what is the top speed of the X-200 engine?"`).
- **Separate agent LLM** means cheap/fast models handle the routing/grading/rewriting calls, while the expensive model is only invoked for the final answer.

### Where the code lives

| File | Purpose |
|------|---------|
| `app/core/agent/planner.py` | Planner agent — routes to `"retrieve"` or `"direct"` |
| `app/core/agent/grader.py` | Context relevance grader |
| `app/core/agent/rewriter.py` | Query rewriter |
| `app/core/agent/tools.py` | `RetrievalTool` — thin wrapper around retriever |
| `app/core/agent/state.py` | `AgentState` dataclass for tracking loop state |
| `app/core/pipeline.py` | Orchestrates the full loop |

---

## 3. Architecture Diagram

<!-- TODO: Insert architecture diagram here -->

*The diagram above illustrates the complete flow: the provider abstraction layer feeding into the agentic RAG loop, with ChromaDB as the vector store and FastAPI as the entry point.*

---

## 4. Coming Soon — Reciprocal Rank Fusion (RRF)

RRF is a **post-retrieval fusion technique** that combines multiple ranked lists into a single, more robust ranking. The idea is straightforward: for each document across all ranked lists, its score is `1 / (k + rank)`, where `k` is a constant (typically 60). Summing these scores across lists produces a fused ranking.

### Why RRF matters for this project

Currently, the retriever does a single **vector search** (HNSW — ChromaDB's default index) per query. This captures semantic similarity well, but can miss exact keyword matches that a lexical approach would catch.

The plan is to add a **BM25 retriever** as a complementary search path:

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Vector search (HNSW)** | Semantic understanding, handles synonyms, works across languages | Needs embeddings, can drift on rare/domain terms |
| **BM25 (lexical)** | Exact keyword matching, fast, zero setup — no embeddings needed | Misses synonyms, bag-of-words, no semantic understanding |

With both retrievers available, RRF becomes the **fusion layer** that combines their ranked lists into a single, more robust result set:

```
Original query ──┬──► Vector search (HNSW, ChromaDB) ──┐
                 │                                       │
                 └──► BM25 lexical search ──────────────┤
                                                        │
                                       ┌────────────────▼──┐
                                       │  RRF Fusion       │
                                       │  1 / (k + rank)   │
                                       └────────┬──────────┘
                                                │
                                                ▼
                                          Grader evaluates
                                          fused context
```

This hybrid approach gives us the best of both worlds: semantic understanding from vectors + precise keyword matching from BM25, fused via RRF into a single ranked list for the grader.

### Implementation plan

1. Add a `post_retrieval` module under `app/core/retrieval/`
2. Implement `rrf_fuse(results: list[list[SourceDocument]], k: int = 60) -> list[SourceDocument]`
3. Add `Bm25Retriever` under `app/core/retrieval/` using a simple inverted index (no external dependencies needed — just tokenization + IDF scoring)
4. Run both retrievers in parallel during the retrieval step and fuse results via RRF
5. Integrate into `pipeline.py` after the retrieval step, before grading
6. Make it configurable (enabled/disabled, retriever weights, k constant) via `config.py`

This is a **pure algorithmic improvement** — no new API calls, minimal dependencies. It directly improves retrieval recall by combining semantic + lexical signals at near-zero latency cost.
