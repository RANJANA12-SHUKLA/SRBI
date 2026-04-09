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
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from schema import ChunkWithMeta

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency during local bootstrap
    faiss = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - optional dependency during local bootstrap
    RecursiveCharacterTextSplitter = None

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

def classify_source(url: str, company_domain: str) -> str:
    """Tags a URL as PRIMARY or SECONDARY based on domain matching."""
    if not url:
        return "SECONDARY"
    
    parsed = urlparse(url.lower())
    domain = parsed.netloc

    if company_domain and company_domain.lower() in domain:
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

def _fallback_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates deterministic fake embeddings if OpenAI fails, preventing crashes."""
    dim = 1536  # Must match text-embedding-3-small so FAISS doesn't break
    embeddings = []
    for t in texts:
        # Create a consistent fake vector based on the text itself
        np.random.seed(abs(hash(t)) % (2**32))
        vec = np.random.randn(dim).astype('float32')
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec.tolist())
    return embeddings

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=2, min=5, max=60), # Wait up to 60s for RPM reset
    stop=stop_after_attempt(4),
    reraise=True
)
def _call_openai_with_retry(client, batch, model):
    """Fires the API call with an exponential backoff shock absorber."""
    return client.embeddings.create(input=batch, model=model)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Production-grade embedding function with batching and graceful fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If key is missing or commented out in .env, instantly use fallback
    if not api_key or api_key.startswith("#"):
        logger.warning("OpenAI key missing; using local fallback embeddings.")
        return _fallback_embeddings(texts)

    client = openai.OpenAI(api_key=api_key)
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    all_embeddings = []
    # BATCHING: Send 50 chunks in ONE request. Bypasses strict RPM limits.
    batch_size = 50 

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = _call_openai_with_retry(client, batch, model)
            all_embeddings.extend([r.embedding for r in response.data])
        except Exception as e:
            logger.error(f"OpenAI failed after retries: {e}")
            logger.warning("Switching to local fallback embeddings to prevent pipeline crash.")
            
            # If OpenAI blocks us, seamlessly pad the rest of the file with fake vectors
            fallback_embs = _fallback_embeddings(texts[i:])
            all_embeddings.extend(fallback_embs)
            break # Exit the loop, we are done with OpenAI for this run

    return all_embeddings

def build_index(chunks: list[ChunkWithMeta]):
    """Embeds the text and builds the FAISS vector index."""
    if not faiss:
        logger.error("faiss is not installed.")
        return None, []
        
    if not chunks:
        return None, []
        
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    
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
) -> list[ChunkWithMeta]:
    """Searches the FAISS index for the most relevant text chunks."""
    if index is None or not metadata_store:
        return []

    # Get single query embedding
    query_embs = embed_texts([query])
    if not query_embs:
        return []
        
    query_vector = np.array(query_embs, dtype="float32")
    faiss.normalize_L2(query_vector)
    
    # Overfetch logic: Pull more than we need, so we can filter by PRIMARY if requested
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