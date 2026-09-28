"""Turn a scored benchmark run into the exact two-column submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd
from .fusion import geo_bias_top50

from .evaluate import load_parts, rank_group


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", choices=["A", "B"], required=True)
    p.add_argument("--geo", choices=["geo_off", "geo_exact"], required=True)
    p.add_argument("--fill-geo-off", action="store_true",
                   help="Keep local top 50 and fill remaining slots from globally ranked results")
    p.add_argument("--geo-bias", action="store_true",
                   help="Fuse both fully reranked pools with the tuned local score bonus")
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    queries = pd.read_parquet(root / "benchmark_queries.parquet", columns=["query_id"])
    items = pd.read_parquet(root / "benchmark_items.parquet", columns=["item_id"])
    scored = load_parts(root / "artifacts" / "cache" / "reranked" /
                        f"benchmark_{args.geo}_{args.branch}")
    groups = {qid: g for qid, g in scored.groupby("eval_id", sort=False)}
    # The published final configuration is sufficient to reproduce branch B
    # without carrying experiment reports into a fresh submission checkout.
    submission_path = root / "config" / "submission.json"
    submission = (json.loads(submission_path.read_text(encoding="utf-8"))
                  if submission_path.is_file() else None)
    if submission and args.branch == submission["branch"] and args.geo == submission["geo"]:
        alpha = float(submission["alpha"])
    else:
        config = json.loads((root / "artifacts" / "experiments" /
                             f"{args.branch}_{args.geo}.json").read_text(encoding="utf-8"))
        alpha = float(config["alpha"])
    global_groups = None
    local_quota = 50
    if args.fill_geo_off or args.geo_bias:
        if args.geo != "geo_exact":
            p.error("geography fusion requires --geo geo_exact")
        if args.fill_geo_off and args.geo_bias:
            p.error("choose one geography fusion method")
        global_scored = load_parts(root / "artifacts" / "cache" / "reranked" /
                                   f"benchmark_geo_off_{args.branch}")
        global_groups = {qid: g for qid, g in global_scored.groupby("eval_id", sort=False)}
        if args.fill_geo_off:
            fill_config = json.loads((root / "artifacts" / "experiments" /
                                      f"{args.branch}_geo_fill.json").read_text(encoding="utf-8"))
            local_quota = int(fill_config["local_quota"])
    bias = None
    if args.geo_bias:
        if args.branch != "B":
            p.error("--geo-bias is defined for direct reranking branch B")
        if submission and args.branch == submission["branch"] and args.geo == submission["geo"]:
            bias = float(submission["geo_bias"])
        else:
            bias_config = json.loads((root / "artifacts" / "experiments" /
                                      "B_geo_bias.json").read_text(encoding="utf-8"))
            bias = float(bias_config["bias"])
    rows = []
    for qid in queries.query_id:
        g = groups.get(qid)
        ids = [] if g is None else items.item_id.iloc[rank_group(g, args.branch, alpha).item_index.astype(int)].tolist()
        if bias is not None:
            gg = global_groups.get(qid)
            ids = items.item_id.iloc[geo_bias_top50(gg, g, bias)].tolist()
        elif global_groups is not None:
            local_ids = ids
            ids = local_ids[:local_quota]
            present = set(ids)
            gg = global_groups.get(qid)
            if gg is not None:
                more = items.item_id.iloc[rank_group(gg, args.branch, alpha, 1000).item_index.astype(int)].tolist()
                for item_id in more:
                    if len(ids) >= 50:
                        break
                    if item_id not in present:
                        ids.append(item_id)
                        present.add(item_id)
            for item_id in local_ids[local_quota:]:
                if len(ids) >= 50:
                    break
                if item_id not in present:
                    ids.append(item_id)
                    present.add(item_id)
        rows.append({"query_id": qid, "answer": " ".join(ids)})
    answer = pd.DataFrame(rows)
    target = root / "answer.csv"
    # Explicit CRLF makes the submission byte-identical on Windows and Linux.
    answer.to_csv(target, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL,
                  lineterminator="\r\n")
    assert answer.query_id.is_unique and len(answer) == len(queries)
    assert answer.query_id.map(len).eq(16).all()
    valid = set(items.item_id)
    for value in answer.answer:
        tokens = value.split()
        assert len(tokens) <= 50 and len(tokens) == len(set(tokens))
        assert all(x in valid and re.fullmatch(r"[0-9a-f]{16}", x) for x in tokens)
    print(f"Validated {target}: {len(answer)} queries")


if __name__ == "__main__":
    main()
