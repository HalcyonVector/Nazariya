# Event clustering evaluation

Generated 2026-10-19 16:35 UTC by `python -m nazariya.eval.clustering`.

> **PROVISIONAL.** 378 of 399 pair labels are LLM silver labels; 21 are human. Pairwise metrics only; B-cubed/AMI need the gold clustering from the final labelling session.

Replays the production online algorithm (`nazariya.cluster.assign_online`, model `bge-m3`, 72 h window) over all 20490 finished articles published in the labelled window. A pair is predicted *same event* iff both articles land in the same event.

| τ | precision | recall | F1 | en–hi recall | events | singletons | events with ≥3 outlets | largest event |
|---|---|---|---|---|---|---|---|---|
| 0.550 | 0.596 | 0.796 | 0.682 | 0.709 | 2621 | 66% | 443 | 982 |
| 0.575 | 0.684 | 0.805 | 0.740 | 0.782 | 4475 | 70% | 624 | 457 |
| 0.600 | 0.744 | 0.850 | 0.793 | 0.818 | 6372 | 73% | 746 | 429 |
| 0.625 | 0.836 | 0.858 | 0.847 | 0.855 | 8059 | 75% | 846 | 453 |
| 0.650 **←** | 0.872 | 0.841 | 0.856 | 0.818 | 9685 | 78% | 871 | 447 |
| 0.675 | 0.946 | 0.770 | 0.849 | 0.727 | 10904 | 79% | 870 | 395 |
| 0.700 | 0.943 | 0.735 | 0.826 | 0.691 | 11905 | 81% | 880 | 337 |
| 0.725 | 0.948 | 0.646 | 0.768 | 0.545 | 12814 | 83% | 854 | 294 |
| 0.750 | 0.969 | 0.558 | 0.708 | 0.473 | 13713 | 85% | 781 | 257 |
| 0.775 | 0.961 | 0.434 | 0.598 | 0.218 | 14550 | 86% | 714 | 230 |
| 0.800 | 0.976 | 0.354 | 0.519 | 0.145 | 15303 | 87% | 665 | 195 |
| 0.825 | 1.000 | 0.283 | 0.441 | 0.091 | 16120 | 89% | 612 | 155 |
| 0.850 | 1.000 | 0.204 | 0.338 | 0.036 | 16871 | 90% | 530 | 126 |

**Best F1 at τ = 0.650** (P 0.872, R 0.841). Production currently uses τ = 0.643.

Notes: article-to-centroid similarity runs higher than article-to-article similarity for true members, so the best clustering τ need not equal the pairwise threshold from the embedding comparison. Watch the largest-event column: a sudden jump means events are chaining together.
