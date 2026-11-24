# RAG Production App

Upload PDFs, embed them into Qdrant, and ask questions about them. Ingestion and
querying run as [Inngest](https://www.inngest.com/) functions behind a FastAPI
app; a Streamlit UI drives both.

## Stack

- **FastAPI + Inngest**: durable workflows (`rag/ingest_pdf`, `rag/query_pdf_ai`)
- **Qdrant**: vector store (cosine, 3072 dims for `text-embedding-3-large`)
- **OpenAI**: embeddings and answer generation
- **Streamlit**: upload and Q&A UI

## Setup

```bash
uv sync
cp .env.example .env   # then set OPENAI_API_KEY
```

`.env`:

```
OPENAI_API_KEY=sk-...
# optional
CHAT_MODEL=gpt-4o-mini
QDRANT_URL=http://localhost:6333
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

## Run (4 terminals)

```bash
# 1. Qdrant
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# 2. FastAPI app serving the Inngest functions
uv run uvicorn main:app --reload

# 3. Inngest dev server
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

# 4. UI
uv run streamlit run streamlit_app.py
```
