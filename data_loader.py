import os
from functools import lru_cache

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from openai import OpenAI

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = 3072  # dimension of text-embedding-3-large
EMBED_BATCH_SIZE = 100

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Create the OpenAI client lazily so importing this module never needs an API key."""
    return OpenAI()


def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks: list[str] = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches to stay under the API's per-request input limits."""
    client = get_openai_client()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings
