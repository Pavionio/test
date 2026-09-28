"""Small shared helpers, including the exact task metric."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import psutil

SEARCH_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
    "item_category_id", "item_microcat_id", "item_price", "item_rating",
    "item_rating_reviews_count", "item_location_id", "item_latitude",
    "item_longitude", "item_is_phone_hidden", "item_is_message_forbidden",
]

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokens(value: object) -> list[str]:
    """Tokenize Russian and Latin service text, preserving numbers."""
    return TOKEN_RE.findall(str(value or "").lower().replace("ё", "е"))


def normalize_query(value: object) -> str:
    return " ".join(tokens(value))


def query_text(row: dict) -> str:
    # The search filter is part of the user intent, but has less importance than
    # the actual search phrase. The query is repeated deliberately for retrieval.
    phrase = str(row["search_query"] or "").strip()
    filt = str(row.get("search_infm_params_text", "") or "").strip()
    return f"{phrase}. {phrase}. {filt}" if filt else phrase


def item_text(row: dict, *, dense: bool = False) -> str:
    """Keep title first: it must survive a model's token limit."""
    title = str(row.get("item_title_raw", "") or "")
    params = str(row.get("item_infm_params_text", "") or "")
    desc = str(row.get("item_description_raw", "") or "")
    if dense:
        return f"{title}. {params[:500]}. {desc[:900]}"
    return f"{title} {params[:800]} {desc[:1200]}"


def rerank_text(row: dict) -> str:
    title = str(row.get("item_title_raw", "") or "")
    params = str(row.get("item_infm_params_text", "") or "")
    desc = str(row.get("item_description_raw", "") or "")
    return f"Заголовок: {title}. Параметры: {params[:400]}. Описание: {desc[:950]}"


def recall_at_k(predictions: list[str], relevant: list[str], k: int = 50) -> float:
    positives = set(relevant)
    return len(set(predictions[:k]) & positives) / len(positives) if positives else 0.0


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def peak_memory_bytes() -> int:
    """Peak resident working set on Windows; current RSS elsewhere."""
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def save_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    if not len(scores):
        return np.empty(0, dtype=np.int64)
    k = min(k, len(scores))
    ix = np.argpartition(scores, -k)[-k:]
    return ix[np.argsort(scores[ix])[::-1]]

