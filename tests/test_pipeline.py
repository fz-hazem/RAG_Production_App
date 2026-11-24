import json

import inngest
import pytest
from inngest.experimental import mocked
from qdrant_client import QdrantClient

import main
import vector_db
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult

DIM = 4


def fake_embed(texts):
    """Deterministic fake embeddings: one axis per keyword."""
    keywords = ["qdrant", "inngest", "openai", "streamlit"]
    return [[float(k in t.lower()) for k in keywords] for t in texts]


@pytest.fixture(autouse=True)
def isolated_backends(monkeypatch):
    """In-memory Qdrant + fake embeddings so tests need no network or API key."""
    monkeypatch.setattr(vector_db, "QdrantClient", lambda url, timeout: QdrantClient(":memory:"))
    storage = vector_db.QdrantStorage(dim=DIM)
    monkeypatch.setattr(main, "get_storage", lambda: storage)
    monkeypatch.setattr(main, "embed_texts", fake_embed)
    return storage


@pytest.fixture
def sample_pdf(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(50, 800, "Qdrant stores vectors.")
    c.showPage()
    c.drawString(50, 800, "Inngest runs durable workflows.")
    c.showPage()
    c.save()
    return path


@pytest.mark.parametrize(
    "obj",
    [
        RAGChunkAndSrc(chunks=["a"], source_id="x.pdf"),
        RAGUpsertResult(ingested=3),
        RAGSearchResult(contexts=["c"], sources=["s"]),
    ],
)
def test_pydantic_step_outputs_survive_json_roundtrip(obj):
    """Inngest persists step outputs as JSON; without a Pydantic serializer on the
    client this raises 'Object of type ... is not JSON serializable'. The mocked
    Inngest client used below skips this layer, so it is checked explicitly here."""
    out = main.inngest_client._serialize(obj, type(obj))
    restored = main.inngest_client._deserialize(json.loads(json.dumps(out)), type(obj))
    assert restored == obj


def test_load_pdf_extracts_text(sample_pdf):
    result = main.load_pdf(str(sample_pdf), "sample.pdf")
    assert result.source_id == "sample.pdf"
    assert "Qdrant" in " ".join(result.chunks)


def test_ingestion_is_idempotent(isolated_backends):
    payload = RAGChunkAndSrc(chunks=["Qdrant one", "Inngest two"], source_id="doc.pdf")
    main.embed_and_store(payload)
    main.embed_and_store(payload)
    assert isolated_backends.client.count(isolated_backends.collection).count == 2


def test_empty_pdf_ingests_nothing():
    assert main.embed_and_store(RAGChunkAndSrc(chunks=[], source_id="e.pdf")).ingested == 0


def test_search_returns_best_match_first():
    main.embed_and_store(
        RAGChunkAndSrc(chunks=["Qdrant stores vectors", "Inngest runs workflows"], source_id="d.pdf")
    )
    found = main.search_contexts("tell me about inngest", top_k=1)
    assert found.contexts == ["Inngest runs workflows"]
    assert found.sources == ["d.pdf"]


def test_answer_without_context_does_not_call_llm(monkeypatch):
    def boom():
        raise AssertionError("LLM must not be called without context")

    monkeypatch.setattr(main, "get_openai_client", boom)
    answer = main.generate_answer("anything?", RAGSearchResult(contexts=[], sources=[]))
    assert "couldn't find" in answer


def test_ingest_function_end_to_end(sample_pdf):
    client = mocked.Inngest(app_id="rag_app")
    event = inngest.Event(
        name="rag/ingest_pdf",
        data={"pdf_path": str(sample_pdf), "source_id": "sample.pdf"},
    )
    result = mocked.trigger(main.rag_ingest_pdf, event, client)
    assert result.status == mocked.Status.COMPLETED, result.error
    assert result.output["ingested"] >= 1


def test_query_function_end_to_end(monkeypatch):
    main.embed_and_store(RAGChunkAndSrc(chunks=["Qdrant stores vectors"], source_id="d.pdf"))
    monkeypatch.setattr(main, "generate_answer", lambda q, found: f"answer from {len(found.contexts)} ctx")

    client = mocked.Inngest(app_id="rag_app")
    event = inngest.Event(name="rag/query_pdf_ai", data={"question": "what is qdrant?", "top_k": 3})
    result = mocked.trigger(main.rag_query_pdf_ai, event, client)

    assert result.status == mocked.Status.COMPLETED, result.error
    assert RAGQueryResult(**result.output).sources == ["d.pdf"]
    assert result.output["answer"] == "answer from 1 ctx"
