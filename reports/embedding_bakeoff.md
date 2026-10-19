# Embedding model comparison (event pairs)

Generated 2026-10-19 15:46 UTC by `python -m nazariya.eval.embeddings`.

> **PROVISIONAL.** 378 of 399 labels are LLM silver labels (gpt-oss-120b); 21 are human. Re-run after the human gold labelling session; the `gold` column is the one to trust then.

Task: does cosine similarity of (headline + first 200 words) embeddings separate *same event* pairs from the rest? 399 pairs, 113 same-event, 119 English–Hindi.

| model | ROC-AUC (95% CI) | AUC en–hi pairs | AUC gold only | best-F1 τ | P / R at τ | GPU time (all docs) | CPU docs/s |
|---|---|---|---|---|---|---|---|
| `bge-m3` | 0.982 (0.97–0.99) | 0.979 (0.96–0.99) | 0.956 (0.84–1.00) | 0.643 | 0.94 / 0.90 | 40.5s for 720 | 1.5 |
| `e5-small` | 0.951 (0.93–0.97) | 0.969 (0.94–0.99) | 0.956 (0.83–1.00) | 0.897 | 0.87 / 0.79 | 32.4s for 720 | 14.9 |
| `e5-base` | 0.935 (0.91–0.96) | 0.971 (0.94–0.99) | 0.882 (0.71–1.00) | 0.886 | 0.76 / 0.83 | 19.4s for 720 | 4.5 |
| `mpnet` | 0.915 (0.89–0.94) | 0.889 (0.83–0.94) | 0.971 (0.89–1.00) | 0.670 | 0.72 / 0.79 | 22.3s for 720 | 5.0 |

## Reading this
- **Highest AUC: `bge-m3`.** Its CI overlaps `e5-small`'s (by 0.001), so the lead is suggestive, not established.
- **τ** is the similarity above which a pair is called the same event at best F1; it seeds the clustering threshold. Thresholds are model-specific (E5 similarities are compressed near 1).
- **Cost on CI:** at 1.5 docs/s on this CPU, ~3k new articles/day take ~34.1 CPU-minutes per day for `bge-m3` (GitHub runners are slower than this PC).

## Known biases
- Pairs were sampled from **e5-small** nearest neighbours (plus 50 random pairs). Same-event pairs that e5-small ranks as dissimilar are under-represented, which can flatter e5-small's recall.
- Silver labels come from one LLM; where it is wrong, every model is scored against the same error.
- Only titles + ledes are embedded; body text beyond ~200 words is ignored by design.
