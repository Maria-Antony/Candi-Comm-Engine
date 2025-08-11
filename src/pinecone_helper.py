from typing import List, Dict, Any, Iterable, Optional, Tuple
import time
from datetime import datetime, timedelta, timezone
from pinecone import Pinecone  
import re

from settings import (
    PINECONE_API_KEY, PINECONE_INDEX, PINECONE_HOST,
    INTEGRATED_BATCH_LIMIT, PINECONE_RERANK_MODEL, RETENTION_DAYS
)

# ---------------- Connection / host resolution ----------------
def _client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")
    return Pinecone(api_key=PINECONE_API_KEY)

def _resolve_index_host(pc: Pinecone) -> str:
    if PINECONE_HOST:
        return PINECONE_HOST
    desc = pc.describe_index(name=PINECONE_INDEX)
    host = getattr(desc, "host", None) or (desc.get("host") if isinstance(desc, dict) else None)
    if not host:
        raise RuntimeError(f"Could not resolve host for index '{PINECONE_INDEX}'.")
    return host

def get_index():
    pc = _client()
    try:
        if not pc.has_index(PINECONE_INDEX):
            raise RuntimeError
    except Exception:
        try:
            names = [it["name"] if isinstance(it, dict) else str(it) for it in (pc.list_indexes() or [])]
        except Exception:
            names = []
        raise RuntimeError(f"Index '{PINECONE_INDEX}' not found. Available: {names or '[]'}")
    host = _resolve_index_host(pc)
    return pc, pc.Index(host=host)

def validate_index() -> Dict[str, Any]:
    pc, idx = get_index()
    if not hasattr(idx, "search"):
        raise RuntimeError("Pinecone SDK too old. Please `pip install 'pinecone>=5.0.0'`.")
    return {"ok": True, "name": PINECONE_INDEX}

# ---------------- Utilities ----------------
def chunked(iterable: Iterable[Dict[str, Any]], size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def now_ms() -> int:
    return int(time.time() * 1000)

def ms_ago(days: int) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

# ---------------- Upsert (Integrated Inference) ----------------
def upsert_text_records(records: List[Dict[str, Any]], *, namespace: str) -> int:
    """
    Each record MUST look like:
      {"id": "<unique>", "text": "<full resume text>", ...extra top-level fields}
    """
    if not records:
        return 0
    pc, idx = get_index()
    total = 0
    for batch in chunked(records, INTEGRATED_BATCH_LIMIT):
        norm = []
        for r in batch:
            item = dict(r)
            if "_id" in item and "id" not in item:
                item["id"] = item.pop("_id")
            if "text" not in item:
                item["text"] = item.pop("content", "")
            norm.append(item)
        idx.upsert_records(namespace=namespace, records=norm)  # <— Pinecone embeds these server-side
        total += len(norm)
    return total

# ---------------- Search (Integrated) ----------------
def _search_candidates(
    query_text: str,
    *,
    namespace: str,
    top_k: int,
    fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Plain semantic search using the index's hosted embedding model."""
    pc, idx = get_index()
    params: Dict[str, Any] = {
        "namespace": namespace,
        "query": {"inputs": {"text": query_text}, "top_k": int(top_k)},
    }
    if fields:
        params["fields"] = fields
    return idx.search(**params)

def _normalize_hits(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits = result.get("result", {}).get("hits", []) or result.get("matches", []) or []
    out: List[Dict[str, Any]] = []
    for h in hits:
        rid  = h.get("id") or h.get("_id")
        sc   = float(h.get("score") or h.get("_score") or 0.0)
        flds = h.get("fields") or h.get("metadata") or {}
        txt  = flds.get("text") or flds.get("chunk_text") or flds.get("content") or ""
        name = flds.get("name") or flds.get("candidate_name") or rid
        out.append({"id": rid, "score": sc, "text": txt, "metadata": flds, "name": name})
    return out

# ---------------- Re-rank (Inference API) ----------------
def _inference_rerank(query_text: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
    """
    Uses Pinecone Inference API (pc.inference.rerank).
    Returns list of dicts with {'index', 'score'} and (optionally) 'document.text'.
    """
    pc = _client()
    model = PINECONE_RERANK_MODEL or "pinecone-rerank-v0"
    res = pc.inference.rerank(
        model=model,
        query=query_text,
        documents=documents,
        top_n=int(top_n),
        return_documents=True,
    )

    items = []
    for r in res.data:
        # support both attr and dict styles just in case
        idx = getattr(r, "index", None)
        if idx is None and isinstance(r, dict):
            idx = r.get("index")
        score = getattr(r, "score", None) if idx is not None else None
        doc = getattr(r, "document", None)
        text = getattr(doc, "text", "") if doc is not None else ""
        items.append({"index": int(idx), "rerank_score": float(score), "text": text})
    return items

# ---------------- Public: search -> rerank -> normalized rows ----------------
def search_and_rerank_text(
    query_text: str,
    namespace: str,
    top_k: int,
    fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    1) search with integrated embeddings (same model as index)
    2) rerank those hits via Pinecone Inference (e.g., pinecone-rerank-v0)
    Returns hits ordered by rerank score, each with both 'score' and 'rerank_score'.
    """
    # 1) candidate search
    raw = _search_candidates(query_text, namespace=namespace, top_k=top_k, fields=fields)
    hits = _normalize_hits(raw)
    if not hits:
        return []

    # 2) rerank using documents' text (order preserved by 'index')
    docs = [h["text"] or "" for h in hits]
    rr = _inference_rerank(query_text, docs, top_n=top_k)

    # 3) stitch back to hits
    ranked: List[Dict[str, Any]] = []
    for item in rr:
        i = item["index"]
        base = hits[i]
        ranked.append({**base, "rerank_score": item["rerank_score"]})
    # already in descending order via the API; if not, sort:
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked

def wipe_namespace(*, namespace: str) -> Tuple[int, Optional[str]]:
    pc, idx = get_index()
    try:
        for namespace in idx.list_namespaces():
            res = idx.delete_namespace(namespace=namespace.name)
        deleted = int(res.get("num_deleted", -1)) if isinstance(res, dict) else -1
        return deleted, None
    except Exception as e:
        return -1, str(e)
