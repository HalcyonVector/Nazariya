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

## Ethics
- Respects robots.txt for every fetch.
- Identifies itself as `NazariyaBot`, except for 4 outlets (News18, Firstpost, PIB, ANI) whose CDN
  blocks every non-browser UA; robots.txt is still obeyed for them.
- Never bypasses paywalls.
- Publishes no article text: only metadata, embeddings, derived labels and links. Extracted text is
  kept in a 14-day processing cache (`article_text`) so models can be re-run, and is never exported.
