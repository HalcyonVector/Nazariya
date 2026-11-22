# Nazariya — Project Plan

A "Ground News for India": ingest 30–40 Indian outlets daily (English, Hindi, regional),
cluster articles into events across languages, and analyse how each outlet covers and
frames the same events. Free-tier only, transparent methods, real evaluation metrics.

## Principles
1. **Static-first.** Pipeline runs on GitHub Actions cron; frontend is a static Next.js site
   reading JSON snapshots. Nothing needs to stay awake.
2. **Distillation.** LLM → silver labels → human verification → small fine-tuned multilingual
   models (ONNX, CPU) in production.
3. **"No data" ≠ "didn't cover".** Feed outages / paywalls are tracked explicitly and excluded
   from blindspot statistics.
4. **No republishing.** Article text lives only in memory during a run. We store metadata,
   embeddings, derived labels, and link out.
5. **Free tiers only, budgeted per month.** There is no budget for paid plans, so nothing here may
   depend on one. Every recurring job states what it costs in each free quota (Neon: ~100 compute-hours,
   0.5 GB storage and a data-transfer cap per month, so verify in its console since limits change;
   Groq: daily token caps; GitHub Actions: free for public repos), and cadence and retention are chosen
   from that arithmetic. Jobs cap their own work so a quota cannot be silently exhausted. A paid upgrade
   is never the fallback.

## Architecture
```
GitHub Actions (cron, every 4h)
  fetch → extract (trafilatura) → dedup/syndication → embed → cluster → NLP → aggregate → export JSON
     │                                     │
  fetch_runs (health)        Postgres+pgvector (Neon/Supabase, hot ~7d: ~12k articles/day is ~20-25 MB/day, estimated) + Parquet on HF Datasets (cold)
Kaggle/Colab GPU: fine-tune → ONNX int8 → HF Hub     HF Space: Argilla (labelling)     Groq: silver labels/claims
Frontend: Next.js static export on Vercel / GitHub Pages
```

## Components
| Component | Choice | Notes |
|---|---|---|
| Fetch | httpx + feedparser; adapter chain RSS → news sitemap → Google News RSS → GDELT | per-source fallbacks |
| Extract | trafilatura (favor_recall) | fallback to RSS description, `access` flag |
| Dedup | canonical URL + MinHash/SimHash | also detects PTI/ANI/IANS syndicated copy |
| Embeddings | multilingual-e5-base (title + first ~200 words) | bake-off vs bge-m3, mpnet on own pair set |
| Clustering | online centroid assignment (72h window, τ + entity check) + nightly HDBSCAN merge/split | two levels: event ⊂ story |
| Entities | GLiNER multilingual + Wikidata alias table | QIDs unify English/Hindi names |
| Framing | MFC 15 frames, multi-label, India codebook; MuRIL / XLM-R fine-tune | SemEval-2023 T3 transfer |
| Stance | entity-targeted 3-class pair classifier | separate event sentiment / quotes / outlet stance |
| Claims | LLM extraction + mDeBERTa-xnli matching within event | only events with ≥3 outlets |
| Blindspots | log-odds w/ informative Dirichlet prior ("Fightin' Words") | volume-normalised, "no data" excluded |
| Fingerprints | coverage / frame / stance vectors → CA/PCA + bootstrap hierarchical clustering + Procrustes over time | language axis shown explicitly |
| Labelling | Argilla on HF Space, LLM pre-annotations, active learning | blind test set labelled w/o suggestions |

## Evaluation
| Component | Gold data | Metrics |
|---|---|---|
| Extraction | 60 hand-extracted pages | token P/R/F1 |
| Dedup | 200 pairs | P/R |
| Embeddings | 400 same/different-event pairs (~30% EN–HI) | ROC-AUC, cross-lingual AUC, τ |
| Clustering | ~500 articles, 3-day window | B-cubed P/R/F1, AMI, ARI, cross-lingual recall, fragmentation |
| Framing | ~400 verified + ~120 blind test | macro-F1, per-language/outlet F1, α agreement |
| Stance | ~600 (article, entity) pairs | macro-F1, confusion matrix |
| Aggregates | — | adjusted classify-and-count, bootstrap CIs, ECE |
| Claims | 30 articles, 150 pairs | faithfulness precision, recall, matching F1 |
| Blindspots | — | weekly precision@10, bootstrap/τ sensitivity |
| Fingerprints | — | bootstrap ARI, Procrustes disparity |

### Labelling strategy (revised 2026-10-19)
- **Build first, label last.** Every component is built and evaluated against LLM silver labels
  (gpt-oss-120b primary; local Qwen 3.5 9B as an independent second opinion). Reports mark these
  numbers **provisional**, and every eval script reruns unchanged once gold labels exist.
- **One human labelling session at the end**, covering all components, kept small: binary
  questions only, only pairs where the two LLM labellers disagree, plus a blind random sample
  to estimate the error rate where they agree.
- **Event pairs are binary** (same event / not). A first pass with a third "same story" class
  failed: the LLMs read it as "same topic". Story-level grouping will instead link events through
  shared named entities and time, with its own definition ("same named case or incident").

## Free-tier budget (storage)
Neon's free plan (~100 compute-hours a month) ran out twice: in ~3.5 days in October with hourly runs,
and again by mid-November at one run every 4 hours. A hosted database with a monthly compute cap cannot
carry this pipeline, so since 2026-11-23 there is no database server:
- **State**: one SQLite file. Each Actions run downloads it from the repo's `state` release, works
  locally and uploads it again (`scripts/state.sh`), encrypted with the STATE_KEY secret because release
  assets of a public repo are public and the file holds a few days of article text. Integrity-checked
  before every upload, plus one rotating backup per weekday. No quota; Actions minutes are free for
  public repos.
- **Size**: raw rows 8 days, article text 3 days (`nazariya/retain.py`), about 60 MB. The fact tables
  the models read (event_facts, coverage, copies, title_edits) are kept indefinitely: on the order of
  10 MB a month, so a year stays well inside a release asset's 2 GB limit.
- **Home PC**: keeps its own SQLite file and hands new articles to the cloud as an encrypted
  `home_inbox` asset (`nazariya/inbox.py`); only one writer ever touches the main state.

## Analysis (revised 2026-11-23)
The site is a set of models of how outlets work, computed every run from the fact tables
(`nazariya/analytics/`): under-covered events, outlet map (correspondence analysis), lead-lag,
selection model (logistic regression on first-report features), attention model (Poisson on population
and distance from Delhi), story half-life (Kaplan-Meier), wire dependence, outlet-to-outlet copying,
headline edits, and English-Hindi drift. Framing and stance stay as inputs, not as the product.

## Risks
Scraper breakage & IP blocks · feed outages → false blindspots · paywalls (never bypass) ·
syndication inflation · language/region confound · stance-label conceptual errors ·
labelling cost · free-tier limits (DB 500 MB, Groq quotas, Supabase idle pause) ·
political/legal sensitivity (methods page, error bars, neutral axis names) · drift.

## Timeline
- **W1 Ingestion:** registry, feed probe, fetch→extract→dedup→Postgres, Actions cron, health table; label event pairs; embedding bake-off.
- **W2 MVP:** online clustering, JSON export, Next.js feed with coverage strips + health page; clustering gold set & metrics.
- **W3 Entities & labelling:** GLiNER + Wikidata, codebook, Argilla, LLM pre-annotation, blind test set.
- **W4 Classifiers:** TF-IDF baseline → zero-shot LLM → fine-tuned MuRIL/XLM-R; ONNX; frames in UI.
- **W5 Claims & blindspots:** claim extraction + NLI, Fightin' Words, weekly report, scale to 30–40 outlets, GDELT backfill.
- **W6 Fingerprints & hardening:** outlet map + timeline, methods page, model cards, failure drills.

Cut first if behind: regional languages → claims → animated map.
