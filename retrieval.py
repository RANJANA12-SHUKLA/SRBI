from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from openai import OpenAI, APIError, AuthenticationError, BadRequestError, RateLimitError

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
    import tiktoken
except Exception:  # pragma: no cover - optional dependency during local bootstrap
    tiktoken = None

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
)

_OPENAI_CLIENT: OpenAI | None = None
_EMBEDDING_DISABLED_REASON: str | None = None
_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}


class EmbeddingUnavailableError(RuntimeError):
    pass


def _embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def _embedding_dimensions() -> int | None:
    raw_value = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid EMBEDDING_DIMENSIONS value: %s", raw_value)
        return None


def _embedding_batch_item_limit() -> int:
    raw_value = os.getenv("EMBEDDING_BATCH_MAX_ITEMS", "128").strip()
    try:
        return max(1, min(int(raw_value), 2048))
    except ValueError:
        return 128


def _embedding_batch_token_limit() -> int:
    raw_value = os.getenv("EMBEDDING_BATCH_MAX_TOKENS", "120000").strip()
    try:
        return max(1, min(int(raw_value), 300000))
    except ValueError:
        return 120000


def _embedding_max_attempts() -> int:
    raw_value = os.getenv("EMBEDDING_MAX_ATTEMPTS", "2").strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 2


@lru_cache(maxsize=4)
def _encoding_for_embeddings():
    if tiktoken is None:
        return None
    model = _embedding_model_name()
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _normalize_embedding_text(text: str) -> str:
    return text.replace("\r", "\n").strip()


def _count_embedding_tokens(text: str) -> int:
    encoding = _encoding_for_embeddings()
    if encoding is None:
        return max(1, len(text) // 4)
    return max(1, len(encoding.encode(text)))


def _truncate_for_embedding(text: str, max_tokens: int = 8191) -> str:
    encoding = _encoding_for_embeddings()
    if encoding is None:
        approx_limit = max_tokens * 4
        return text[:approx_limit]
    encoded = encoding.encode(text)
    if len(encoded) <= max_tokens:
        return text
    return encoding.decode(encoded[:max_tokens])


def _iter_embedding_batches(texts: list[str]) -> list[tuple[list[str], list[int]]]:
    item_limit = _embedding_batch_item_limit()
    token_limit = _embedding_batch_token_limit()

    batches: list[tuple[list[str], list[int]]] = []
    current_batch: list[str] = []
    current_indices: list[int] = []
    current_tokens = 0

    for index, raw_text in enumerate(texts):
        normalized = _truncate_for_embedding(_normalize_embedding_text(raw_text))
        token_count = _count_embedding_tokens(normalized)

        would_overflow_items = len(current_batch) >= item_limit
        would_overflow_tokens = current_batch and (current_tokens + token_count > token_limit)
        if would_overflow_items or would_overflow_tokens:
            batches.append((current_batch, current_indices))
            current_batch = []
            current_indices = []
            current_tokens = 0

        current_batch.append(normalized)
        current_indices.append(index)
        current_tokens += token_count

    if current_batch:
        batches.append((current_batch, current_indices))

    return batches


def _rate_limit_reason(exc: RateLimitError) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            if code and message:
                return f"{code}: {message}"
            if message:
                return str(message)
    return str(exc)


def _is_hard_quota_error(exc: RateLimitError) -> bool:
    haystack = _rate_limit_reason(exc).lower()
    return "insufficient_quota" in haystack or "quota" in haystack

def classify_source(url: str, company_domain: str) -> str:
    """Tags a URL as PRIMARY or SECONDARY based on domain matching."""
    if not url:
        return "SECONDARY"
    
    normalized_url = url.lower()
    parsed = urlparse(normalized_url)
    domain = parsed.netloc

    if company_domain and company_domain.lower() in domain:
        return "PRIMARY"

    if "linkedin.com/company/" in normalized_url:
        return "PRIMARY"
        
    for pattern in PRIMARY_STATIC_PATTERNS:
        if pattern in domain:
            return "PRIMARY"
            
    return "SECONDARY"

def chunk_document(text: str, metadata: dict, chunk_size: int = 500, chunk_overlap: int = 50) -> list[ChunkWithMeta]:
    """Splits raw text into 500-token chunks using Langchain."""
    if not RecursiveCharacterTextSplitter:
        logger.error("langchain_text_splitters not installed.")
        return []
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_text(text)
    
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(ChunkWithMeta(
            text=chunk_text,
            source_url=metadata.get("source_url", metadata.get("url", "")),
            source_type=metadata.get("source_type", "SECONDARY"),
            scraped_at=metadata.get("scraped_at", ""),
            chunk_index=i
        ))
    return chunks

def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("#"):
            raise EmbeddingUnavailableError("OPENAI_API_KEY is not set.")
        _OPENAI_CLIENT = OpenAI(api_key=api_key, max_retries=0)
    return _OPENAI_CLIENT


def _disable_embeddings(reason: str) -> None:
    global _EMBEDDING_DISABLED_REASON
    if _EMBEDDING_DISABLED_REASON is None:
        _EMBEDDING_DISABLED_REASON = reason
        logger.error("Disabling remote embeddings for this run: %s", reason)

def _call_openai_with_retry(client: OpenAI, batch: list[str], model: str):
    request_kwargs: dict[str, object] = {
        "input": batch,
        "model": model,
        "encoding_format": "float",
    }
    dimensions = _embedding_dimensions()
    if dimensions is not None:
        request_kwargs["dimensions"] = dimensions

    max_attempts = _embedding_max_attempts()
    for attempt in range(1, max_attempts + 1):
        try:
            return client.embeddings.create(**request_kwargs)
        except RateLimitError as exc:
            if _is_hard_quota_error(exc) or attempt >= max_attempts:
                raise
            backoff_seconds = min(float(2 ** (attempt - 1)), 8.0)
            logger.warning(
                "Embedding rate limit hit for batch of %d texts; retrying in %.1fs (attempt %d/%d).",
                len(batch),
                backoff_seconds,
                attempt + 1,
                max_attempts,
            )
            time.sleep(backoff_seconds)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with OpenAI and cache results. No local fake fallback is used."""
    if not texts:
        return []

    if _EMBEDDING_DISABLED_REASON is not None:
        raise EmbeddingUnavailableError(_EMBEDDING_DISABLED_REASON)

    model = _embedding_model_name()
    client = _get_openai_client()

    results: list[list[float]] = []
    uncached_texts: list[str] = []
    uncached_indices: list[int] = []
    cached_results: dict[int, list[float]] = {}

    for index, text in enumerate(texts):
        normalized = _truncate_for_embedding(_normalize_embedding_text(text))
        cache_key = (model, normalized)
        if cache_key in _EMBEDDING_CACHE:
            cached_results[index] = _EMBEDDING_CACHE[cache_key]
        else:
            uncached_texts.append(normalized)
            uncached_indices.append(index)

    fetched_vectors: dict[int, list[float]] = {}
    for batch, relative_indices in _iter_embedding_batches(uncached_texts):
        batch_indices = [uncached_indices[index] for index in relative_indices]
        try:
            response = _call_openai_with_retry(client, batch, model)
        except RateLimitError as exc:
            reason = _rate_limit_reason(exc)
            if _is_hard_quota_error(exc):
                _disable_embeddings(reason)
                raise EmbeddingUnavailableError(reason) from exc
            logger.error("Embedding request failed after retries: %s", reason)
            raise EmbeddingUnavailableError(reason) from exc
        except (AuthenticationError, BadRequestError, APIError) as exc:
            _disable_embeddings(str(exc))
            raise EmbeddingUnavailableError(str(exc)) from exc

        for target_index, item, source_text in zip(batch_indices, response.data, batch, strict=False):
            vector = item.embedding
            _EMBEDDING_CACHE[(model, source_text)] = vector
            fetched_vectors[target_index] = vector

    for index in range(len(texts)):
        if index in cached_results:
            results.append(cached_results[index])
        else:
            results.append(fetched_vectors[index])
    return results

def build_index(chunks: list[ChunkWithMeta]):
    """Embeds the text and builds the FAISS vector index."""
    if not faiss:
        logger.error("faiss is not installed.")
        return None, []
        
    if not chunks:
        return None, []
        
    texts = [c.text for c in chunks]
    try:
        embeddings = embed_texts(texts)
    except EmbeddingUnavailableError as exc:
        logger.error("Embedding build failed: %s", exc)
        return None, []
    
    if not embeddings:
        return None, []
        
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    
    # Normalize vectors for Inner Product to act as Cosine Similarity
    embs_array = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embs_array)
    index.add(embs_array)
    
    metadata = [c.model_dump() for c in chunks]
    return index, metadata

def save_index(index, metadata: list[dict], company_id: str, indices_dir: str | Path):
    """Saves the FAISS index and metadata to disk."""
    if not index or not faiss:
        return
        
    dir_path = Path(indices_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(dir_path / f"{company_id}.faiss"))
    with open(dir_path / f"{company_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def load_index(company_id: str, indices_dir: str | Path):
    """Loads a previously built FAISS index from disk."""
    if not faiss:
        return None, None
        
    dir_path = Path(indices_dir)
    faiss_path = dir_path / f"{company_id}.faiss"
    meta_path = dir_path / f"{company_id}_meta.json"
    
    if not faiss_path.exists() or not meta_path.exists():
        return None, None
        
    try:
        index = faiss.read_index(str(faiss_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as exc:
        logger.warning(f"Failed to load persisted index for {company_id}: {exc}")
        return None, None

def retrieve(
    query: str,
    index,
    metadata_store: list[dict],
    top_k: int = 8,
    source_type_filter: Optional[str] = None,
    query_vector: Optional[list[float]] = None,
) -> list[ChunkWithMeta]:
    """Searches the FAISS index for the most relevant text chunks."""
    if index is None or not metadata_store:
        return []

    if query_vector is None:
        try:
            query_embs = embed_texts([query])
        except EmbeddingUnavailableError:
            return []
        if not query_embs:
            return []
        query_array = np.array(query_embs, dtype="float32")
    else:
        query_array = np.array([query_vector], dtype="float32")

    faiss.normalize_L2(query_array)
    
    # Overfetch logic: Pull more than we need, so we can filter by PRIMARY if requested
    overfetch = max(top_k * 3, top_k)
    distances, indices = index.search(query_array, min(overfetch, len(metadata_store)))

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
    """Formats the retrieved chunks so the LLM can read them and cite URLs."""
    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Chunk {idx}]",
                    f"source_url: {chunk.source_url}",
                    f"source_type: {chunk.source_type}",
                    f"text: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(blocks)
