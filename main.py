import asyncio
import logging
import os
import uuid
from functools import lru_cache

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI

from custom_types import (
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
)
from data_loader import EMBED_DIM, embed_texts, get_openai_client, load_and_chunk_pdf
from vector_db import QdrantStorage

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    # Required so steps can return Pydantic models: without it, Inngest cannot
    # JSON-serialize step outputs and every step fails.
    serializer=inngest.PydanticSerializer(),
)


@lru_cache(maxsize=1)
def get_storage() -> QdrantStorage:
    return QdrantStorage(dim=EMBED_DIM)


# --- Pipeline logic -----------------------------------------------------------
# Plain synchronous functions, kept outside the Inngest handlers so they can be
# unit-tested without a running Inngest server.


def load_pdf(pdf_path: str, source_id: str | None = None) -> RAGChunkAndSrc:
    chunks = load_and_chunk_pdf(pdf_path)
    return RAGChunkAndSrc(chunks=chunks, source_id=source_id or pdf_path)


def embed_and_store(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
    chunks = chunks_and_src.chunks
    source_id = chunks_and_src.source_id or "unknown"
    if not chunks:
        return RAGUpsertResult(ingested=0)
    vecs = embed_texts(chunks)
    # Deterministic ids: re-ingesting the same PDF overwrites instead of duplicating.
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunk} for chunk in chunks]
    get_storage().upsert(ids, vecs, payloads)
    return RAGUpsertResult(ingested=len(chunks))


def search_contexts(question: str, top_k: int = 5) -> RAGSearchResult:
    query_vec = embed_texts([question])[0]
    found = get_storage().search(query_vec, top_k)
    return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])


def generate_answer(question: str, found: RAGSearchResult) -> str:
    if not found.contexts:
        return "I couldn't find anything relevant in the ingested documents."
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using only the context above."
    )
    response = get_openai_client().chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You answer questions using only the provided context."},
            {"role": "user", "content": user_content},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# --- Inngest functions --------------------------------------------------------
# The pipeline functions above do blocking I/O (PDF parsing, OpenAI, Qdrant), so
# each step offloads them to a worker thread to keep the event loop responsive.


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context) -> dict:
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", pdf_path)

    async def _load() -> RAGChunkAndSrc:
        return await asyncio.to_thread(load_pdf, pdf_path, source_id)

    async def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        return await asyncio.to_thread(embed_and_store, chunks_and_src)

    chunks_and_src = await ctx.step.run("load-and-chunk", _load, output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run(
        "embed-and-upsert", _upsert, chunks_and_src, output_type=RAGUpsertResult
    )
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> dict:
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    async def _search() -> RAGSearchResult:
        return await asyncio.to_thread(search_contexts, question, top_k)

    async def _answer(found: RAGSearchResult) -> str:
        return await asyncio.to_thread(generate_answer, question, found)

    found = await ctx.step.run("embed-and-search", _search, output_type=RAGSearchResult)
    answer = await ctx.step.run("llm-answer", _answer, found)

    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    ).model_dump()


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])
