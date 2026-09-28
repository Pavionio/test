"""Shared final fusion, used identically in validation and submission."""

from __future__ import annotations

import numpy as np
import pandas as pd


def geo_bias_top50(global_group: pd.DataFrame | None,
                   local_group: pd.DataFrame | None, bias: float) -> np.ndarray:
    """Return item row indices ranked by cross-encoder logit plus local bonus.

    Duplicate ads occur in both pools. Keep one score per item; the reranker
    cache guarantees the same score for a repeated query-item pair.
    """
    score = {} if global_group is None else dict(zip(global_group.item_index,
                                                     global_group.ce_score))
    local_ix = set() if local_group is None else set(local_group.item_index)
    if local_group is not None:
        score.update(zip(local_group.item_index, local_group.ce_score))
    ix = np.fromiter(score.keys(), dtype=np.int64)
    ce = np.fromiter(score.values(), dtype=np.float32)
    is_local = np.fromiter((i in local_ix for i in ix), dtype=np.float32)
    values = ce + bias * is_local
    if len(values) > 50:
        order = np.argpartition(values, -50)[-50:]
        order = order[np.argsort(values[order])[::-1]]
    else:
        order = np.argsort(values)[::-1]
    return ix[order]
