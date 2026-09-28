# Experiment log

Every metric below is query-macro Recall: each distinct complete search request
has equal weight, regardless of how many clicked ads it has. The 400 tuning and
200 branch-selection holdout requests are disjoint by normalized search text from the CatBoost
fit set. The 1400 other held-out requests measure retrieval only. Their clicked
ads remain in the 344,825-ad validation corpus, but their click labels never
enter model fitting before the final refit.

The run began at the UTC time in `artifacts/experiments/run_start.json` and has
a ten-hour computation budget. Installation and the one-time model downloads
are excluded from that limit. Each JSON report and per-query prediction file
in `artifacts/experiments/` is part of the run record. For reproducibility,
`provenance.json` records package versions, the GPU, Qdrant, and input/weight
hashes. The initial pilot scores 50 validation requests before larger runs.
`compute_pauses.json` records one desktop pause from 02:05:57 to 09:59:50 UTC
on 2026-09-28, evidenced by the completed global-run timing file and the
next Python process start time. No experiment process ran during those 7h 54m;
the budget calculation excludes this idle interval when resumed.

## Retrieval, all 2000 held-out requests

| Geography | Recall@1000 | Empty candidate lists | Title channel | Body channel | Character channel | Dense channel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No exact filter | 0.7329 | 0 | 0.3610 | 0.5184 | 0.4460 | 0.6214 |
| Exact location except delivery | 0.7485 | 338 | 0.5020 | 0.6818 | 0.6457 | 0.7438 |

The channel columns mean Recall from that channel's members *within the final
1000-item union*, so they are diagnostic rather than independent retrieval
runs. The exact-location filter is sent to Qdrant and applied to every other
channel. It improves the overall candidate list but leaves 16.9% of these
validation requests without any candidates. The test suite checks every
persisted local candidate against its request's location whenever delivery is
off. The benchmark corpus likewise gives zero exact-local candidates for
427 of 2452 requests; the chosen two-pool fusion still returns global ads for
them.

`search_category` carries little discrimination in the validation sample:
1999 of 2000 requests use category 114, and 344,814 of 344,825 train-corpus
ads have `item_category_id=114`. Benchmark queries include 222 unrestricted
category-0 requests. The CatBoost experiments use category as a soft feature;
the selected direct-rerank pipeline never makes it a hard retrieval filter.

## Reranking, 400 tuning plus 200 holdout requests

| Variant | Tuning Recall@50 | Holdout Recall@50 | Observation |
| --- | ---: | ---: | --- |
| B, no exact filter: direct BGE rerank of 1000 | 0.4058 | 0.3750 | Global dense/lexical candidates are difficult to order by text alone. |
| B, exact filter | 0.7017 | 0.6725 | More useful local set, but 94 of 600 requests are empty. |
| A, no exact filter: CatBoost top 150 then BGE | 0.5838 | 0.6300 | Tuning selected blend coefficient 2.0. |
| A, separately trained exact-filter CatBoost | 0.6398 | 0.6425 | Prefilter Recall@150 is 0.6838; tuning selected BGE-only score within its 150. |
| B, tuned local/global quota | 0.7979 | 0.7325 | Tuning selected 48 local places; global results fill the rest. |
| A, tuned local/global quota | 0.7285 | 0.7013 | Tuning selected 40 local places. |
| B, tuned additive local score bonus | 0.8046 | **0.7500** | Tuning selected +8 to raw BGE logits for ads in exact-local retrieval. |

The initial A/exact-filter attempt used the CatBoost model trained on global
candidates. It had only 0.2642 prefilter Recall@150 and 0.2875 holdout
Recall@50 because the feature distributions changed under exact-local
retrieval. Its report and predictions are retained with the suffix
`_mismatched`; they are **not** used to select the deployed pipeline. The
corrected experiment trains an independent CatBoost model on exact-local
candidates, scores the same 600 validation requests, and then reranks them
with the shared BGE model. The other three strict variants are unchanged.
The corrected prefilter model retrieves 2040 of 2698 known fitting positives
at the 1000-item candidate stage. On the validation requests its 150-item
prefilter retains 0.6838 of the relevant ads; the final top 50 reaches 0.6425.

The optional quota experiment uses the four scored branches but does not
modify their separate results. On the 400 tuning requests, it evaluates fixed
local quotas of 25, 35, 40, 45, 48, and 50 out of 50; it then reports the
chosen quota on the untouched 200 holdout requests. A quota of 50 simply
fills vacant slots. Final selection compares all strict and quota variants
that fit the remaining wall-clock budget.

The measured winner is B with both complete pools and a +8 local score bonus.
Its holdout Recall@50 is **0.7500**. `selection.json` records a conservative estimate of
11,817 seconds for two complete benchmark reranking runs, compared with more
than 26,000 seconds remaining at selection. The first complete benchmark run
was already in progress at that point, so the actual remaining time is lower.
The full benchmark has no public labels; this number is a local holdout
estimate, not a claimed benchmark score.

The final global B run scored 2,452,000 query-item pairs in 5,494 seconds
(446 pairs/s). The exact-location B run processed 1,762,072 pairs; 105,811
scores were already cached from the global run, and scoring the other 1,656,261
took 3,817 seconds (434 new pairs/s). The measured peak Python working set
during the exact-location run was 4.83 GB. Both runs used the saved local
reranker weights. `answer.csv` was produced from the two cached score sets and
independently checked against the original benchmark Parquet files: 2,452
unique 16-character query IDs, exactly 50 distinct valid item IDs per row,
no empty rows, and exactly the required two CSV columns.

The selected fusion has a 0.9325 retrieval ceiling on the 200 holdout
requests: 15 clicked ads are absent from both 1000-item candidate pools, and
39 more are present but miss the final 50. The quota scheme had 42 final
selection misses on the same requests. These counts and each affected query
are saved by `src.error_analysis`.

Because the 200 holdout requests are used to choose among six variants, their
winning score can be optimistic. After locking the choice, `src.evaluate_audit`
checks it on 200 additional requests from the retrieval-only split. These
requests contributed to the aggregate Recall@1000 figure, but they did not
select CatBoost parameters, a reranker blend, geography, or the quota. The
audit logs its own per-query predictions and a bootstrap interval. It cannot
replace the hidden benchmark, but it reveals whether the selected lift
persists outside the branch-selection set.

The completed post-selection audit reached Recall@50 **0.7775** on those 200
queries (2,000-bootstrap 95% interval **0.7200–0.8300**). On the same requests,
direct BGE reranking of global candidates reached 0.37125, and direct
reranking of exact-local candidates reached 0.6450. Exact-local retrieval
returned no candidates for 41 requests; the selected fusion filled all of
them from the global pool. The per-query answers and scores are stored in
`audit_B_200_predictions.parquet`, and the summary is in `audit_B_200.json`.

## Files and provenance

- `artifacts/experiments/retrieval.json`: retrieval metrics, splits, channels,
  empty-list count, and time.
- `artifacts/experiments/{A,B}_{geo_off,geo_exact}.json`: four strict reranking
  reports, each with tuning/holdout metrics and timing.
- `artifacts/experiments/{A,B}_geo_fill.json`: supplemental quota results.
- `artifacts/experiments/*_predictions.parquet` and `*_per_query.csv`: the
  actual validation predictions and query-level scores.
- `artifacts/experiments/selection.json`: final holdout-based choice and time
  budget calculation.
- `artifacts/experiments/resource_sample.json`: an observed Windows process
  peak working set during benchmark B reranking (about 4.8 GB). Later timing
  manifests also record peak process memory directly.
- `artifacts/experiments/*_timing.json`: pair counts, new cross-encoder calls,
  elapsed time, throughput, and peak memory for validation, benchmark, and
  independent audit runs. The same manifests live beside their resumable
  score caches, which are rebuilt during reproduction.
- `artifacts/models/*.cbm`, `models/`: locally saved trained and pretrained
  weights; Git LFS handles binary model files.

All reported quality figures come from locally held-out `train` records.
`benchmark_queries.parquet` has no labels and is never used to calculate or
tune these metrics.
