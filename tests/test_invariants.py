"""Checks that guard the validation protocol and submission format."""

from pathlib import Path

import pandas as pd

from src.common import normalize_query, recall_at_k
from src.fusion import geo_bias_top50


def test_macro_query_recall() -> None:
    assert recall_at_k(["a", "b"], ["a", "c"], 2) == 0.5
    assert recall_at_k([], ["a"], 50) == 0.0


def test_geo_bias_fusion_deduplicates_overlapping_pools() -> None:
    global_ = pd.DataFrame({"item_index": [1, 2], "ce_score": [3.0, 2.0]})
    local = pd.DataFrame({"item_index": [2, 3], "ce_score": [2.0, 1.5]})
    assert geo_bias_top50(global_, local, 2.0).tolist() == [2, 3, 1]


def test_validation_texts_not_in_fit_pairs() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not (root / "validation_queries.parquet").exists():
        return
    val = pd.read_parquet(root / "validation_queries.parquet", columns=["search_query", "relevant"])
    fit = pd.read_parquet(root / "fit_pairs.parquet", columns=["search_query"])
    assert not set(val.search_query.map(normalize_query)) & set(fit.search_query.map(normalize_query))
    assert val.relevant.map(len).ge(1).all()


def test_validation_positives_exist_in_search_corpus() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not (root / "validation_queries.parquet").exists():
        return
    val = pd.read_parquet(root / "validation_queries.parquet", columns=["relevant"])
    corpus = set(pd.read_parquet(root / "train_items.parquet", columns=["item_id"]).item_id)
    assert all(set(ids).issubset(corpus) for ids in val.relevant)


def test_exact_geo_candidates_obey_the_filter() -> None:
    """Check the actual merged output, including BM25 and Qdrant hits."""
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts" / "cache" / "candidates" / "validation_geo_exact"
    files = sorted(folder.glob("part_*.parquet"))
    if not files:
        return
    queries = pd.read_parquet(root / "data" / "processed" / "validation_queries.parquet",
                              columns=["eval_id", "search_location_id", "search_is_delivery_search"])
    qmap = queries.set_index("eval_id")
    item_locations = pd.read_parquet(root / "data" / "processed" / "train_items.parquet",
                                     columns=["item_location_id"]).item_location_id.to_numpy()
    for part in files:
        candidates = pd.read_parquet(part, columns=["eval_id", "item_index"])
        if candidates.empty:
            continue
        q = qmap.loc[candidates.eval_id]
        local = q.search_is_delivery_search.to_numpy() == 0
        assert (item_locations[candidates.item_index.to_numpy()[local]] ==
                q.search_location_id.to_numpy()[local]).all()
