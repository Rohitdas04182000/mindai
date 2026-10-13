# MindAI — Graph-RAG Chatbot for GitHub Repositories

MindAI is a codebase question-answering assistant that uses Retrieval-Augmented Generation (RAG) to answer natural-language questions about GitHub repositories with verifiable, line-level source citations.

It combines tree-sitter AST-aware code chunking, Qdrant-native hybrid retrieval, Reciprocal Rank Fusion (RRF), reranking, conditional call-graph expansion, and LangGraph workflow orchestration.

The system validates generated citations against the retrieved source context. If validation fails, it retries once and then refuses rather than returning a response with unsupported source references.

## Key Features

- **AST-aware code chunking:** Uses tree-sitter to identify meaningful code structures such as functions and classes.
- **Hybrid retrieval:** Combines dense vector search with BM25-style sparse search using Qdrant.
- **Reciprocal Rank Fusion:** Merges dense and sparse rankings without directly combining incomparable raw scores.
- **Reranking:** Uses Cohere reranking when configured, with a local sentence-transformers CrossEncoder fallback.
- **Conditional graph expansion:** Uses caller/callee relationships to retrieve additional code for trace and call-chain questions when initial retrieval appears insufficient.
- **Citation validation:** Rejects citations to files or line ranges outside the retrieved context and retries once before refusing.
- **RAG evaluation:** Provides DeepEval scripts for faithfulness, answer relevancy, and contextual recall.
- **Automated tests:** Includes a 59-test offline pytest suite covering retrieval fusion, citation validation, input validation, and graph-expansion routing.
- **Web application:** Provides a FastAPI backend and React frontend with authentication, repository management, chat history, and server-sent events (SSE).

## Architecture

```text
                 REPOSITORY INGESTION
                          |
                          v
                   GitHub Repository
                          |
                          v
                   Shallow Clone
                          |
                          v
                   File Filtering
                          |
                          v
                 tree-sitter / AST
                          |
                          v
                  AST-aware Chunking
                          |
                          v
                   Code + Metadata
                          |
                    +-----+-----+
                    |           |
                    v           v
              Dense Embedding  Sparse Embedding
              OpenAI API       FastEmbed
                    |           |
                    +-----+-----+
                          |
                          v
                   Qdrant Collection
                 Named Dense + Sparse Vectors
                          |
                          v
                 QUESTION-ANSWERING
                          |
                          v
                    User Question
                          |
                          v
                    Query Rewrite
                          |
                          v
                 Hybrid Retrieval
                   /           \
                  v             v
             Dense Search    Sparse Search
                   \           /
                    v         v
                      RRF Fusion
                          |
                          v
                       Reranking
                          |
                          v
                Conditional Graph Expansion
                          |
                          v
                   Context Assembly
                          |
                          v
                    LLM Generation
                          |
                          v
                 Citation Validation
                    /           \
               Valid             Invalid
                 |                  |
                 v                  v
             Final Answer      One Retry
                                    |
                                    v
                           Refusal if Still Invalid
```

## How It Works

### 1. Repository Ingestion

The ingestion pipeline shallow-clones the target repository, filters out irrelevant files and credential files, and processes supported source-code languages.

Tree-sitter parses source code into an Abstract Syntax Tree (AST). AST-aware chunking uses code structure to create meaningful retrieval units rather than relying exclusively on arbitrary fixed-size text windows.

Each chunk carries metadata such as its file path, line range, symbol information, and source content. This metadata enables the system to associate retrieved evidence with its original location.

For languages and file types that cannot be parsed by tree-sitter, the system uses line-based fallback chunking.

### 2. Embeddings and Indexing

Each code chunk is represented using two types of vectors:

- **Dense vectors:** Generated using OpenAI's `text-embedding-3-small` embedding model for semantic retrieval.
- **Sparse vectors:** Generated locally using FastEmbed's BM25-style sparse embedding model for lexical retrieval.

Both named vectors and their associated payload metadata are stored on the same Qdrant point.

Each repository receives its own Qdrant collection, keeping its indexed code separate from other repositories.

### 3. Hybrid Retrieval and RRF

When a user asks a question, MindAI rewrites the query and performs dense and sparse searches against Qdrant.

Dense retrieval helps locate code based on semantic similarity, while sparse retrieval helps find exact identifiers, function names, constants, and keywords.

The two ranked lists are combined using Reciprocal Rank Fusion:

\[
\operatorname{RRF}(d)=
\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)}
\]

Here, \(d\) is a document, \(R\) represents the ranked result lists, and \(k\) is a smoothing constant.

RRF combines results according to their ranks rather than adding raw dense and sparse scores, which operate on different scales.

The implementation retrieves up to 100 candidates from each search, fuses their rankings, deduplicates the results, and passes the candidate chunks to reranking.

### 4. Reranking

The reranking stage evaluates the relevance of retrieved code chunks to the rewritten question.

MindAI supports two reranking paths:

- **Cohere:** Uses the hosted Cohere Rerank API when `MINDAI_COHERE_API_KEY` is configured.
- **Local fallback:** Uses the `cross-encoder/ms-marco-MiniLM-L-6-v2` CrossEncoder when Cohere is not configured.

The reranked candidates are used to construct the context supplied to the LLM.

### 5. Conditional Call-Graph Expansion

For questions that involve tracing execution or understanding caller/callee relationships, the graph can expand the retrieved context using extracted call-graph relationships.

Expansion is conditional rather than unconditional: the query must resemble a trace or call-chain question, and the initial retrieval must appear insufficient.

The call graph is extracted using source-code pattern matching. It is therefore approximate and cannot resolve every dynamic call, reflective invocation, or dependency-injection relationship.

### 6. Answer Generation

The final generation stage receives the user's question and the retrieved code context.

The LLM uses that evidence to produce a natural-language explanation. The generation instructions require source citations for factual claims and encourage refusal when the available evidence does not support a useful answer.

The source metadata allows citations to identify files and line ranges.

### 7. Citation Validation

Citation validation is performed after generation.

The validator checks:

- Whether the cited file is represented in the retrieved context.
- Whether the cited line range overlaps a line range retrieved for that file.
- Whether the citation follows the accepted source-reference format.

If validation fails, MindAI retries once with the valid retrieved paths available to the generation stage. If the retry still fails validation, the system returns a refusal instead of the unsupported response.

**Important limitation:** Citation validation checks source provenance and line-range overlap. It does not independently prove that the cited lines semantically support every claim. Faithfulness evaluation is used to assess grounding at the answer level.

## Technology Stack

| Component | Technology |
|---|---|
| Workflow orchestration | LangGraph |
| Language model and dense embeddings | OpenAI API |
| Vector database | Qdrant |
| Sparse embeddings | FastEmbed |
| Code parsing | tree-sitter |
| Ranking fusion | Reciprocal Rank Fusion |
| Reranking | Cohere Rerank API / sentence-transformers CrossEncoder |
| RAG evaluation | DeepEval |
| Backend | FastAPI |
| Frontend | React, Vite, Tailwind CSS |
| Authentication and persistence | JWT, SQLAlchemy, SQLite by default |
| Testing | pytest |

## Project Structure

```text
mindai/
├── ingestion.py             # Repository ingestion and indexing
├── graph.py                 # LangGraph RAG workflow
├── dense_embeddings.py      # Dense embedding provider
├── reranker.py              # Cohere reranking integration
├── semantic_chunking.py     # Alternative semantic chunking strategy
├── eval.py                  # DeepEval metrics and standalone mock evaluation
├── eval_live.py             # Evaluation against an indexed repository
├── eval_hard.py             # Evaluation with source-derived reference answers
├── tests/
│   └── test_pipeline.py     # Offline pipeline tests
├── backend/
│   ├── main.py              # FastAPI application and routes
│   ├── auth.py              # Authentication
│   ├── db.py                # Database configuration
│   ├── models.py            # Persistence models
│   ├── schemas.py            # Request/response schemas
│   └── pipeline_store.py    # Pipeline and repository state
├── frontend/
│   ├── src/
│   └── package.json
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python installed
- Node.js and npm installed
- An OpenAI API key for the embedding and generation calls
- An optional Cohere API key for hosted reranking

### 1. Clone the Repository

```bash
git clone https://github.com/Rohitdas04182000/mindai.git
cd mindai
```

### 2. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the repository root.

At minimum, configure:

```dotenv
OPENAI_API_KEY=your_openai_api_key
MINDAI_JWT_SECRET=your_secure_random_secret
```

Generate a random JWT secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Optional configuration:

| Variable | Purpose |
|---|---|
| `QDRANT_URL` | Qdrant server or cloud URL |
| `QDRANT_API_KEY` | Authentication for hosted Qdrant |
| `MINDAI_QDRANT_PATH` | Local Qdrant storage directory |
| `DATABASE_URL` | Database connection string; defaults to SQLite |
| `MINDAI_GENERATION_MODEL` | Model used for answer generation |
| `MINDAI_REWRITE_MODEL` | Model used for query rewriting |
| `MINDAI_JUDGE_MODEL` | Model used by DeepEval judges |
| `MINDAI_COHERE_API_KEY` | Enables hosted Cohere reranking |
| `MINDAI_CHUNK_STRATEGY` | Selects `ast` or `semantic` chunking |
| `MINDAI_JUDGE_SPACING` | Delay between evaluation judge calls |

Changing the chunking strategy requires re-ingesting the repository.

### 4. Run the Core Pipeline

Ingest a repository:

```bash
python ingestion.py https://github.com/owner/repository
```

Ask a question:

```bash
python graph.py https://github.com/owner/repository "How does authentication work?"
```

Run the standalone mock evaluation:

```bash
python eval.py
```

**Note:** The command-line question-answering path re-ingests the repository before answering. It is intended for one-off pipeline inspection. For repeated questions against one index, use the web application or the live evaluation script.

### 5. Run the Backend

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Start the FastAPI application:

```bash
python -m uvicorn backend.main:app --reload --port 8001
```

### 6. Run the Frontend

Open a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser, register or sign in, ingest a repository, and start asking questions.

The application uses server-sent events to send status updates and the completed answer to the frontend. The current implementation does not stream individual LLM tokens as they are generated.

## Testing

MindAI includes a pytest suite covering core pipeline behavior.

Run the tests with:

```bash
python -m pytest tests/test_pipeline.py -q
```

The documented test run reports 59 passing tests. The suite covers:

- GitHub URL validation.
- Credential-file filtering.
- RRF scoring and result deduplication.
- Citation validation, including fabricated paths and line ranges.
- Context-size limits.
- Conditional graph-expansion routing.
- Chunking-related constraints.

These are offline tests and do not require live OpenAI, Cohere, or Qdrant services.

## Evaluation

MindAI includes three evaluation entry points:

```bash
python eval.py
python eval_live.py
python eval_hard.py
```

- `eval.py` defines the DeepEval metrics and includes a standalone mock dataset.
- `eval_live.py` evaluates the pipeline against an already-indexed repository.
- `eval_hard.py` uses reference answers written from repository source code before the evaluation, reducing the risk of circular ground truth.

The three metrics are:

| Metric | What it measures |
|---|---|
| Faithfulness | Whether the generated answer is supported by the retrieved context |
| Answer Relevancy | Whether the answer addresses the user's question |
| Contextual Recall | Whether the retrieved context contains the information needed for the reference answer |

### Reported Evaluation Results

The project CV reports the following DeepEval scores on a 705-chunk codebase:

| Metric | Reported score |
|---|---:|
| Faithfulness | 0.967 |
| Answer Relevancy | 0.987 |
| Contextual Recall | 0.917 |

These are run-specific scores, not a guarantee of performance on every repository or question.

**Reproducibility note:** The repository audit documents that a fresh end-to-end evaluation after subsequent code changes could not be completed because the OpenAI account exhausted its available credits. Re-run the evaluation scripts against the intended indexed collection to establish scores for the current implementation. Do not treat the historical scores as a verified benchmark of later changes.

The evaluation process also exposed an important methodology issue: reference answers written after observing the system's output can inflate contextual recall. The hard evaluation script instead uses reference answers derived directly from repository source.

LLM-judged metrics can also be sensitive to judge-model choice and long contexts. Evaluation scores should therefore be interpreted alongside test coverage, retrieval behavior, and manual inspection of representative answers.

## Performance and Engineering Considerations

The project CV reports reducing repository-ingestion latency from approximately 48 seconds to under 20 seconds using hosted embeddings and parallel processing.

Dense embeddings are generated through the OpenAI API, with concurrent batches. Hosted Cohere reranking is used when configured; otherwise, reranking falls back to the local CrossEncoder.

Query latency depends on network access and the configured providers. Repository ingestion and query-answering latency are separate measurements.

## Known Limitations

- **Citation validation is not semantic proof.** A real source location may still be interpreted incorrectly by the LLM.
- **Call-graph extraction is approximate.** Regex-based extraction cannot resolve every dynamic call, reflection pattern, or dependency-injection relationship.
- **No incremental re-indexing.** A repository index represents a snapshot. New commits require re-ingestion.
- **Local Qdrant mode limits concurrency.** The backend serializes pipeline access with a process-wide lock when using local on-disk storage. A Qdrant server or cloud deployment is more appropriate for concurrent workloads.
- **SQLite is the default database.** Configure a suitable database such as PostgreSQL for a larger deployment.
- **Fallback chunking is line-based.** Unsupported file formats fall back to line windows rather than full AST parsing.
- **Broad conceptual questions remain challenging.** Questions that require information distributed across many files may not retrieve sufficient evidence and can result in a refusal.
- **Evaluation requires API access and credits.** DeepEval's LLM-based judges can be expensive and sensitive to model choice and context length.
- **Repository freshness is not automatic.** The system does not continuously synchronize an index with new repository commits.

## Future Improvements

Potential improvements include:

- Incremental re-indexing after Git commits.
- More robust call-graph construction using language-aware symbol resolution.
- Better query decomposition for questions spanning multiple files.
- Broader evaluation datasets with source-derived reference answers.
- End-to-end regression evaluation after changes to retrieval, reranking, or generation.
- Improved citation verification that evaluates semantic support in addition to source provenance.
- Better observability for indexing latency, retrieval quality, and per-stage query latency.

## License

See the repository's license file for usage and distribution terms.
