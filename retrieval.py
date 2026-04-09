from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from schema import ChunkWithMeta

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency during local bootstrap
    faiss = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - optional dependency during local bootstrap
    RecursiveCharacterTextSplitter = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency during local bootstrap
    OpenAI = None


logger = logging.getLogger(__name__)

PRIMARY_STATIC_PATTERNS = (
    "bse.com",
    "nseindia.com",
    "mca.gov.in",
    "sebi.gov.in",
)
ALWAYS_SECONDARY_PATTERNS = (
    "techcrunch.com",
    "yourstory.com",
    "inc42.com",
    "theken.com",
    "economictimes.com",
    "moneycontrol.com",
    "crunchbase.com",
    "tracxn.com",
    "pitchbook.com",
    "bloomberg.com",
    "reuters.com",
    "livemint.com",
    "business-standard.com",
    "businessstandard.com",
    "entrackr.com",
    "vccircle.com",
    "dealstreetasia.com",
)
FALLBACK_EMBEDDING_DIMENSION = 1536


class SimpleIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype("float32")

    def search(self, query: np.ndarray, k: int):
        query = query.astype("float32")
        scores = self.vectors @ query.T
        indices = np.argsort(scores[:, 0])[::-1][:k]
        distances = scores[indices, 0].reshape(1, -1)
        return distances, indices.reshape(1, -1)


def _normalize_domain(domain: str) -> str:
    return domain.lower().replace("www.", "").strip()


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return _normalize_domain(parsed.netloc or parsed.path.split("/")[0])


def classify_source(url: str, company_domain: str) -> str:
    normalized_url = url.lower()
    domain = _extract_domain(url)
    company_domain = _normalize_domain(company_domain)

    if company_domain and company_domain in domain:
        return "PRIMARY"

    if "linkedin.com/company/" in normalized_url:
        return "PRIMARY"

    if any(pattern in domain or pattern in normalized_url for pattern in PRIMARY_STATIC_PATTERNS):
        return "PRIMARY"

    if any(pattern in domain or pattern in normalized_url for pattern in ALWAYS_SECONDARY_PATTERNS):
        return "SECONDARY"

    return "SECONDARY"


def chunk_document(
    text: str,
    metadata: dict,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[ChunkWithMeta]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str]
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)
    else:
        step = max(chunk_size - chunk_overlap, 1)
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), step)]

    output: list[ChunkWithMeta] = []
    for index, chunk in enumerate(chunks):
        output.append(
            ChunkWithMeta(
                text=chunk,
                source_url=metadata.get("source_url", ""),
                source_type=metadata.get("source_type", "SECONDARY"),
                scraped_at=metadata.get("scraped_at", ""),
                chunk_index=index,
            )
        )
    return output


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return None
    return OpenAI(api_key=api_key)


def _hash_embedding(text: str, dimension: int = FALLBACK_EMBEDDING_DIMENSION) -> list[float]:
    vector = np.zeros(dimension, dtype="float32")
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return vector.tolist()

    for token in tokens:
        slot = hash(token) % dimension
        vector[slot] += 1.0

    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector.tolist()


@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(5), reraise=True)
def _embed_batch_openai(texts: list[str], model: str) -> list[list[float]]:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client unavailable")

    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    vectors: list[list[float]] = []

    client = _get_openai_client()
    if client is None:
        logger.warning("OpenAI client unavailable; using deterministic local fallback embeddings.")
        return [_hash_embedding(text) for text in texts]

    for start in range(0, len(texts), 100):
        batch = texts[start : start + 100]
        try:
            vectors.extend(_embed_batch_openai(batch, model))
        except Exception as exc:
            logger.warning("Embedding batch failed; falling back to local embeddings: %s", exc)
            vectors.extend(_hash_embedding(text) for text in batch)
    return vectors


def _indices_dir() -> Path:
    return Path(os.getenv("INDICES_DIR", "./indices"))


def build_index(chunks: list[ChunkWithMeta]):
    if not chunks:
        vectors = np.zeros((0, FALLBACK_EMBEDDING_DIMENSION), dtype="float32")
        metadata_store: list[dict] = []
        return SimpleIndex(vectors), metadata_store

    texts = [chunk.text for chunk in chunks]
    vectors = np.array(embed_texts(texts), dtype="float32")
    metadata_store = [chunk.model_dump() for chunk in chunks]

    if faiss is not None:
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return index, metadata_store

    normalized = vectors.copy()
    for idx, row in enumerate(normalized):
        norm = np.linalg.norm(row)
        if norm:
            normalized[idx] = row / norm
    return SimpleIndex(normalized), metadata_store


def save_index(index, metadata: list[dict], company_id: str) -> None:
    directory = _indices_dir()
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / f"{company_id}_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if faiss is not None and hasattr(index, "ntotal"):
        faiss.write_index(index, str(directory / f"{company_id}.faiss"))
        return

    if isinstance(index, SimpleIndex):
        np.save(directory / f"{company_id}.npy", index.vectors)


def load_index(company_id: str):
    directory = _indices_dir()
    meta_path = directory / f"{company_id}_meta.json"
    if not meta_path.exists():
        return None, None

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    faiss_path = directory / f"{company_id}.faiss"
    numpy_path = directory / f"{company_id}.npy"

    try:
        if faiss is not None and faiss_path.exists():
            return faiss.read_index(str(faiss_path)), metadata
        if numpy_path.exists():
            return SimpleIndex(np.load(numpy_path).astype("float32")), metadata
    except Exception as exc:
        logger.warning("Failed to load persisted index for %s: %s", company_id, exc)
        return None, None

    return None, metadata


def retrieve(
    query: str,
    index,
    metadata_store: list[dict],
    top_k: int = 8,
    source_type_filter: Optional[str] = None,
) -> list[ChunkWithMeta]:
    if index is None or not metadata_store:
        return []

    query_vector = np.array(embed_texts([query]), dtype="float32")
    overfetch = max(top_k * 3, top_k)
    distances, indices = index.search(query_vector, min(overfetch, len(metadata_store)))

    results: list[ChunkWithMeta] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata_store):
            continue
        chunk = ChunkWithMeta.model_validate(metadata_store[int(idx)])
        if source_type_filter and chunk.source_type != source_type_filter:
            continue
        results.append(chunk)
        if len(results) >= top_k:
            break
    return results


def format_context_chunks(chunks: list[ChunkWithMeta]) -> str:
    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Chunk {idx}]",
                    f"source_url: {chunk.source_url}",
                    f"source_type: {chunk.source_type}",
                    f"scraped_at: {chunk.scraped_at}",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(blocks)
