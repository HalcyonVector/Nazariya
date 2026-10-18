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
skips them (`--runner cloud`) and a Windows scheduled task on a home PC ingests them every 2 hours:
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
