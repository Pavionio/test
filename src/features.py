"""Numeric CatBoost features; no held-out click labels are inspected here."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .common import tokens
from .retrieve import FEATURES as RETRIEVAL_FEATURES

MODEL_FEATURES = RETRIEVAL_FEATURES + [
    "same_location", "same_category", "item_microcat_id", "item_price",
    "item_rating", "item_rating_reviews_count", "item_is_phone_hidden",
    "item_is_message_forbidden", "title_overlap", "title_all", "body_overlap",
    "query_words", "has_filter", "delivery", "title_length",
]


class FeatureBuilder:
    def __init__(self, root: Path, dataset: str, query_dataset: str):
        self.items = pd.read_parquet(root / "data" / "processed" / f"{dataset}_items.parquet")
        queries = pd.read_parquet(root / "data" / "processed" / f"{query_dataset}_queries.parquet")
        self.queries = {r["eval_id"]: r for r in queries.to_dict("records")}
        # Scoring millions of pairs through DataFrame.iloc is unnecessarily
        # expensive. Keep column arrays and tokenize each query only once.
        self.columns = {name: self.items[name].to_numpy() for name in (
            "item_location_id", "item_category_id", "item_microcat_id",
            "item_price", "item_rating", "item_rating_reviews_count",
            "item_is_phone_hidden", "item_is_message_forbidden", "item_title_raw",
            "item_infm_params_text", "item_description_raw")}
        self.query_tokens = {qid: set(tokens(row["search_query"]))
                             for qid, row in self.queries.items()}

    @lru_cache(maxsize=200000)
    def item_tokens(self, ix: int) -> tuple[set[str], set[str]]:
        title = set(tokens(self.columns["item_title_raw"][ix]))
        body = set(tokens(str(self.columns["item_infm_params_text"][ix])[:500] + " " +
                          str(self.columns["item_description_raw"][ix])[:700]))
        return title, body

    def build(self, candidates: pd.DataFrame) -> pd.DataFrame:
        output = candidates[RETRIEVAL_FEATURES].copy()
        extra = []
        for c in candidates.itertuples(index=False):
            query = self.queries[c.eval_id]
            ix = int(c.item_index)
            qtok = self.query_tokens[c.eval_id]
            title, body = self.item_tokens(ix)
            qt = max(len(qtok), 1)
            col = self.columns
            price = col["item_price"][ix]
            rating = col["item_rating"][ix]
            reviews = col["item_rating_reviews_count"][ix]
            extra.append((
                int(int(query["search_location_id"]) == int(col["item_location_id"][ix])),
                # Search category 0 means an unrestricted search, not an item
                # category; it must not penalize service-category listings.
                int(int(query["search_category"]) == 0 or
                    int(query["search_category"]) == int(col["item_category_id"][ix])),
                float(col["item_microcat_id"][ix]),
                float(price) if pd.notna(price) else -1.0,
                float(rating) if pd.notna(rating) else -1.0,
                float(reviews) if pd.notna(reviews) else 0.0,
                int(bool(col["item_is_phone_hidden"][ix])),
                int(bool(col["item_is_message_forbidden"][ix])),
                len(qtok & title) / qt, int(bool(qtok) and qtok.issubset(title)),
                len(qtok & body) / qt, len(qtok),
                int(bool(query["search_infm_params_text"])),
                int(query["search_is_delivery_search"]),
                len(str(col["item_title_raw"][ix])),
            ))
        names = MODEL_FEATURES[len(RETRIEVAL_FEATURES):]
        output[names] = pd.DataFrame(extra, columns=names, index=candidates.index)
        return output[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(-1)

