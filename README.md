# Nazariya

How do Indian outlets cover the same events? An open, transparent coverage and framing
analyser for ~40 English, Hindi and regional news outlets. See [PLAN.md](PLAN.md).

## Setup
```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"     # Windows; use .venv/bin/pip on Linux/macOS
```
Without `DATABASE_URL` everything runs against a local SQLite file (`data/nazariya.db`).
Set `DATABASE_URL` to a Postgres connection string (Neon/Supabase) and install `.[pg]` to use Postgres.

## Commands
```bash
python -m nazariya.probe --write-feeds   # discover/verify sources → config/feeds.yaml, reports/feed_probe.md
python -m nazariya.ingest                # discover → filter → extract → dedup (one run)
python -m nazariya.health                # per-outlet health → reports/health.md
python -m nazariya.cluster               # embed new articles (bge-m3) and assign them to events
python -m nazariya.entities.run          # link people/parties/states/institutions in new articles
python -m nazariya.export                # JSON snapshots for the static site → site/data/
python -m nazariya.eval.embeddings       # embedding model comparison → reports/embedding_bakeoff.md
python -m nazariya.eval.clustering       # τ sweep for clustering → reports/clustering_eval.md
pytest -q
```

## Config
- `config/outlets.yaml`: hand-curated outlet registry (no leaning labels). `user_agent: browser`
  marks the few outlets whose CDN rejects any bot user agent; `rewrite_hosts` fixes bad feed links.
- `config/feeds.yaml`: generated per-outlet source chain (sitemap / RSS → Google News fallback).
- `config/filters.yaml`: out-of-scope URL sections (sport, entertainment, astrology, media pages…).

## Pipeline (week 1)
1. **Discover**: try each outlet's source chain in order; keep only in-scope links (host + homepage
   path, because BBC/Zee sitemaps list every edition); drop excluded sections; every attempt is
   logged in `fetch_runs`.
2. **Extract**: newest-first, up to 100 articles per outlet per run, robots.txt-checked, trafilatura.
   Pending items older than 48h are marked `skipped`.
3. **Dedup**: 64-bit SimHash. Same outlet ≤3 bits → `duplicate`; other outlet ≤6 bits → `syndicated_of`.
   Wire bylines (PTI/ANI/IANS/UNI/Reuters/AFP, incl. Hindi forms) → `agency`.
4. **Dates**: future timestamps (IST labelled as UTC, e.g. Dainik Bhaskar) are shifted and flagged.

## Events and the site (week 2)
5. **Embed**: every finished article from the last 72 h (body lede when we have it, otherwise headline
   + feed summary, so headline-only outlets still count as coverage), with `bge-m3` (best of four
   multilingual models on English–Hindi same-event pairs). Vectors are kept 5 days.
6. **Cluster**: online, centroid-based. In publication order, each article joins the most similar event
   active within 72 h if cosine similarity ≥ τ = 0.65 (best pairwise F1 on labelled pairs), otherwise it
   starts a new event. Large running stories (e.g. a stampede and its aftermath) end up as one event.
7. **Export → site**: `site/` is a static page (no build step) reading `site/data/*.json`. Each story
   shows a coverage strip, one cell per outlet: covered, wire copy only, not covered, or *no data*
   (no successful fetch from that outlet during the story's window). Clusters where one outlet
   supplies most articles (templated local items) are left out.

8. **Entities** (week 3): a dictionary linker over a Wikidata alias table of ~1,500 entities (Indian
   politicians, parties, states/UTs, key institutions, a few countries and foreign leaders) with English
   and Hindi names. Hindi spellings are normalised (गाँधी = गांधी, नरेन्द्र = नरेंद्र); short acronyms
   count only for parties/institutions and only in capitals; a surname like "Modi" goes to someone
   named in full in the article, or to a clearly more prominent candidate, or nowhere. Single-word
   names that are mostly ordinary words (पूजा, गंभीर, प्रधान…) are dropped by a corpus calibration.
   ```bash
   python -m nazariya.entities.build       # rebuild config/entities/entities.jsonl from Wikidata
   python -m nazariya.entities.calibrate   # recompute config/entities/alias_calibration.yaml
   python -m nazariya.eval.entities        # LLM-judged precision/recall → reports/entities_eval.md
   ```
   All three outputs are committed so every link can be traced. A neural NER model (GLiNER) was in the
   original plan; it was replaced because it would be as slow on CI CPUs as the embedding model.

9. **Frames** (weeks 3–4): each article's angle in Media Frames Corpus categories, adapted for Indian
   news ([labels/FRAMING_CODEBOOK.md](labels/FRAMING_CODEBOOK.md)). Silver labels from LLMs on 1,500
   articles from multi-outlet stories; four classifiers compared in
   [reports/framing_eval.md](reports/framing_eval.md). Production uses logistic regression on the
   bge-m3 vectors clustering already stores, so framing costs almost nothing and covers headline-only
   outlets too. Fine-tuned XLM-R scores about the same but needs a 1 GB model on CI.
   ```bash
   python -m nazariya.framing.silver --labeller local_long   # LLM silver labels (resumable)
   python -m nazariya.framing.train                          # train + compare → models/framing_bge_lr.npz
   python -m nazariya.framing.infer                          # frame recent articles
   ```

Preview locally: `python -m nazariya.export` then `python -m http.server -d site 8000`.
Publishing is opt-in: enable GitHub Pages (source: GitHub Actions) and set the repository variable
`PAGES_ENABLED=true`; the ingest workflow then deploys the site after every run.

All evaluation numbers are **provisional** until the human labelling session at the end of the build
(see PLAN.md, "Labelling strategy").

## Labelling (event pairs)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml]"
python -m nazariya.label.sample      # once: 400 stratified pairs + 20 repeats → labels/event_pairs_v1.jsonl
python -m nazariya.label.serve       # local page at http://127.0.0.1:8765, keys 1/2/3/0, Z = undo
```
Read [labels/GUIDELINES.md](labels/GUIDELINES.md) first. Pair and label files hold only ids/URLs and
are committed; article text is read from the database at runtime. Texts of labelled articles are
kept past the 14-day cache so evaluations stay reproducible (still never published).

## Home runner
Indian Express, News18, Firstpost, Jansatta and Organiser block datacenter IPs, so GitHub Actions
skips them (`--runner cloud`) and a Windows scheduled task on a home PC ingests them every 4 hours:
```powershell
# one-time: put the Neon connection string in .env (gitignored):  DATABASE_URL=postgresql://...
powershell -ExecutionPolicy Bypass -File scripts\install_home_task.ps1   # re-run after moving the folder
```
Logs go to `data/logs/`. While the PC is off these outlets show as DOWN (= "no data") in the health report.

## Ethics
- Respects robots.txt for every fetch.
- Identifies itself as `NazariyaBot`, except for 4 outlets (News18, Firstpost, PIB, ANI) whose CDN
  blocks every non-browser UA; robots.txt is still obeyed for them.
- Never bypasses paywalls.
- Publishes no article text: only metadata, embeddings, derived labels and links. Extracted text is
  kept in a 14-day processing cache (`article_text`) so models can be re-run, and is never exported.
