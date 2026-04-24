# ESG RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot specialized in ESG (Environmental, Social, Governance) data analysis and reporting.

Built with LangChain, FastAPI, Docker, and ChromaDB — deployable on GCP.

---

##  Purpose

ESG analysts and compliance teams spend hours manually searching through regulatory documents, sustainability reports, and financial filings. This chatbot enables natural language querying over ESG corpora, returning grounded, cited answers with source traceability.

**Key use cases:**
- Query ESG regulatory frameworks (CSRD, SFDR, GRI standards)
- Analyze company sustainability reports
- Cross-reference ESG scores with financial data
- Audit trail for compliance teams

---

##  Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        User Query                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│              /chat  /ingest  /health                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  RAG Pipeline (LangChain)                │
│                                                          │
│  1. Query Analysis & Rewriting                           │
│  2. Hybrid Retrieval (semantic + keyword)                │
│  3. Reranking (CrossEncoder)                             │
│  4. Context Assembly + Grounding                         │
│  5. LLM Generation (OpenAI / Mistral)                    │
│  6. Citation Injection                                   │
└──────────────┬─────────────────────┬────────────────────┘
               │                     │
               ▼                     ▼
┌──────────────────────┐  ┌─────────────────────────────┐
│   ChromaDB           │  │   Document Ingestion         │
│   Vector Store       │  │   Pipeline                   │
│   (Embeddings)       │  │   PDF / CSV / JSON           │
└──────────────────────┘  └─────────────────────────────┘
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| RAG Framework | LangChain |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (local) / Qdrant (production) |
| Reranking | CrossEncoder (`ms-marco-MiniLM`) |
| LLM | OpenAI GPT-4o / Mistral (via API) |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | GCP (Cloud Run + GCS) |
| Observability | LangSmith (tracing) + Prometheus metrics |

---

##  Project Structure

```
esg-rag-chatbot/
│
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── routers/
│   │   ├── chat.py          # /chat endpoint
│   │   └── ingest.py        # /ingest endpoint
│   └── schemas.py           # Pydantic models
│
├── rag/
│   ├── pipeline.py          # Main RAG chain
│   ├── retriever.py         # Hybrid retrieval logic
│   ├── reranker.py          # CrossEncoder reranking
│   ├── embeddings.py        # SentenceTransformers wrapper
│   └── prompts.py           # Prompt templates
│
├── ingestion/
│   ├── loader.py            # PDF / CSV / JSON loaders
│   ├── chunker.py           # Chunking strategies
│   └── pipeline.py          # Full ingestion pipeline
│
├── vectorstore/
│   └── chroma_store.py      # ChromaDB client wrapper
│
├── ui/
│   └── app.py               # Streamlit interface
│
├── evaluation/
│   ├── ragas_eval.py        # RAGAS evaluation framework
│   └── test_queries.json    # Golden test set
│
├── monitoring/
│   └── metrics.py           # Prometheus metrics export
│
├── tests/
│   ├── test_api.py
│   ├── test_rag_pipeline.py
│   └── test_ingestion.py
│
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

##  Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API key (or Mistral)

### 1. Clone & configure

```bash
git clone https://github.com/your-username/esg-rag-chatbot.git
cd esg-rag-chatbot
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Run with Docker

```bash
docker-compose up --build
```

Services will be available at:
- **API**: http://localhost:8000
- **UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### 3. Ingest documents

```bash
# Ingest ESG documents from the /data directory
curl -X POST http://localhost:8000/ingest \
  -F "files=@data/csrd_guidelines.pdf"
```

### 4. Query the chatbot

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the CSRD reporting requirements for scope 3 emissions?"}'
```

---

##  RAG Pipeline Details

### Ingestion
1. **Loading**: Supports PDF, CSV, JSON via LangChain document loaders
2. **Chunking**: Recursive character splitting (chunk_size=512, overlap=50)
3. **Embedding**: SentenceTransformers `all-MiniLM-L6-v2` (local, no API cost)
4. **Storage**: ChromaDB with metadata filtering (source, date, document_type)

### Retrieval
1. **Semantic search**: cosine similarity on embeddings (top-k=10)
2. **Keyword search**: BM25 over raw text
3. **Hybrid fusion**: Reciprocal Rank Fusion (RRF)
4. **Reranking**: CrossEncoder `ms-marco-MiniLM-L-6-v2` (top-k=4)

### Generation
1. **Context assembly**: top-4 reranked chunks with source metadata
2. **Prompt**: structured with grounding instructions and citation format
3. **Guardrails**: abstention policy when confidence is low
4. **Output**: answer + cited sources + confidence score

---

##  Evaluation

This project uses [RAGAS](https://docs.ragas.io) for automated RAG evaluation:

```bash
python evaluation/ragas_eval.py
```

Metrics tracked:
- **Faithfulness**: Is the answer grounded in retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are retrieved chunks relevant?
- **Context Recall**: Are all relevant chunks retrieved?

---

##  Observability

- **LangSmith**: Full trace of every chain execution (latency, token usage, retrieval quality)
- **Prometheus**: API metrics exposed at `/metrics` (request count, latency, error rate)
- **Structured logging**: JSON logs with correlation IDs for every request

---

## ☁️ GCP Deployment

```bash
# Build and push to GCP Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/esg-rag-chatbot

# Deploy to Cloud Run
gcloud run deploy esg-rag-chatbot \
  --image gcr.io/PROJECT_ID/esg-rag-chatbot \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

---

##  Roadmap

- [x] Basic RAG pipeline with ChromaDB
- [x] FastAPI with /chat and /ingest endpoints
- [x] Docker containerization
- [ ] Hybrid search (BM25 + semantic)
- [ ] CrossEncoder reranking
- [ ] RAGAS evaluation framework
- [ ] LangSmith observability
- [ ] GCP Cloud Run deployment
- [ ] LLM agents for multi-step ESG queries
- [ ] Qdrant migration for production

---

## 🧑‍💻 Author

**Rudy Nzau** — Data Scientist & AI Engineer  
[LinkedIn](https://linkedin.com/in/rudy-nzau) · [GitHub](https://github.com/rudy-nzau)

*Built as part of a transition toward AI/LLM Engineering roles, with a focus on regulated and ESG environments.*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
