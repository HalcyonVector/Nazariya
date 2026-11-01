from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import insert, select

from nazariya import cluster, db, retain
from nazariya.db import (
    PINNED_UNTIL, article_events, article_frames, article_text, article_vectors, articles, events,
    fetch_runs, get_engine, mentions, outlets, runs,
)

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def days(n: float) -> datetime:
    return NOW - timedelta(days=n)


def add(conn, aid, age_days, pinned=False, with_children=True):
    conn.execute(insert(articles).values(
        id=aid, outlet_id="a", url=f"https://x/{aid}", title=f"t{aid}", discovered_at=days(age_days),
        published_at=days(age_days), status="extracted", attempts=1, date_fixed=False))
    if with_children:
        conn.execute(insert(article_text).values(
            article_id=aid, text="body", expires_at=PINNED_UNTIL if pinned else NOW + timedelta(days=1)))
        conn.execute(insert(article_vectors).values(
            article_id=aid, model="m", vec=b"\0\0", text_kind="lede", created_at=days(age_days)))
        conn.execute(insert(article_frames).values(
            article_id=aid, primary_frame="policy", scores={}, model="m", created_at=days(age_days)))


def test_retention_deletes_old_keeps_recent_and_pinned(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    with engine.begin() as conn:
        conn.execute(insert(outlets).values(id="a", name="A"))
        add(conn, 1, 12)                 # old, unpinned: deleted with its children
        add(conn, 2, 12, pinned=True)    # old but pinned (labelled / training sample): kept
        add(conn, 3, 5)                  # recent, text older than TEXT_DAYS: row kept, text dropped
        add(conn, 4, 1)                  # recent: everything kept
        conn.execute(insert(events).values(id=1, first_seen=days(12), last_seen=days(12), n_articles=1, n_outlets=1,
                                           model="m", updated_at=days(12)))
        conn.execute(insert(events).values(id=2, first_seen=days(1), last_seen=days(1), n_articles=1, n_outlets=1,
                                           model="m", updated_at=days(1)))
        conn.execute(insert(article_events).values(article_id=1, event_id=1, method="online", assigned_at=days(12)))
        conn.execute(insert(runs).values(id=1, started_at=days(90)))
        conn.execute(insert(runs).values(id=2, started_at=days(1)))
        conn.execute(insert(fetch_runs).values(run_id=1, outlet_id="a", source_url="u", ok=True, accepted=True,
                                               items_seen=0, items_in_scope=0, items_filtered=0, items_new=0,
                                               fetched_at=days(90)))

    dry = retain.run(dry_run=True, now=NOW, engine=engine)
    assert dry["articles"] == 1 and dry["events"] == 1 and dry["runs"] == 1 and dry["fetch_runs"] == 1
    with engine.connect() as conn:  # a dry run deletes nothing
        assert conn.execute(select(articles.c.id)).scalars().all() == [1, 2, 3, 4]

    out = retain.run(now=NOW, engine=engine)
    assert out["articles"] == 1 and out["old_text"] == 1

    with engine.connect() as conn:
        assert sorted(conn.execute(select(articles.c.id)).scalars().all()) == [2, 3, 4]
        assert sorted(conn.execute(select(article_text.c.article_id)).scalars().all()) == [2, 4]  # 3's text dropped
        assert sorted(conn.execute(select(article_vectors.c.article_id)).scalars().all()) == [2, 3, 4]
        assert sorted(conn.execute(select(article_frames.c.article_id)).scalars().all()) == [2, 3, 4]
        assert conn.execute(select(article_events.c.article_id)).scalars().all() == []  # cascaded via event 1
        assert conn.execute(select(events.c.id)).scalars().all() == [2]
        assert conn.execute(select(runs.c.id)).scalars().all() == [2]
        assert conn.execute(select(fetch_runs.c.id)).scalars().all() == []


def test_embed_new_skips_assigned_articles_and_writes_once(tmp_path, monkeypatch):
    engine = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    calls = []

    def fake_embed(model, texts, batch_size=32, **kw):
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32) / 2

    monkeypatch.setattr(cluster, "embed", fake_embed)
    with engine.begin() as conn:
        conn.execute(insert(outlets).values(id="a", name="A"))
        add(conn, 1, 0.1, with_children=False)  # unassigned, no vector: embedded
        add(conn, 2, 0.1, with_children=False)  # assigned to an event, vector already purged: skipped
        conn.execute(insert(events).values(id=1, first_seen=NOW, last_seen=NOW, n_articles=1, n_outlets=1,
                                           model="m", updated_at=NOW))
        conn.execute(insert(article_events).values(article_id=2, event_id=1, method="online", assigned_at=NOW))

    assert cluster.embed_new(engine, NOW, model="m") == 1
    assert calls == [1]
    with engine.connect() as conn:
        assert conn.execute(select(article_vectors.c.article_id)).scalars().all() == [1]
    assert cluster.embed_new(engine, NOW, model="m") == 0  # nothing left to do


def test_housekeeping_keeps_vectors_still_needed(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    with engine.begin() as conn:
        conn.execute(insert(outlets).values(id="a", name="A"))
        for aid in (1, 2, 3, 4):
            add(conn, aid, 0.5, with_children=False)
            conn.execute(insert(article_vectors).values(article_id=aid, model="m", vec=b"\0\0", text_kind="lede",
                                                        created_at=NOW - timedelta(hours=12)))
        conn.execute(insert(events).values(id=1, first_seen=NOW, last_seen=NOW, n_articles=3, n_outlets=1,
                                           model="m", updated_at=NOW, centroid=b"\x02\0\0", rep_article_id=2))
        for aid in (1, 2, 3):  # 4 stays unassigned
            conn.execute(insert(article_events).values(article_id=aid, event_id=1, method="online", assigned_at=NOW))
        for aid in (1, 2):     # 3 is assigned but not framed yet
            conn.execute(insert(article_frames).values(article_id=aid, primary_frame="policy", scores={}, model="m",
                                                       created_at=NOW))
    assert retain.run(now=NOW, engine=engine)["used_vectors"] == 1
    with engine.connect() as conn:
        # 1: assigned+framed -> purged; 2: representative of an open event -> kept; 3: unframed -> kept; 4: unassigned -> kept
        assert sorted(conn.execute(select(article_vectors.c.article_id)).scalars().all()) == [2, 3, 4]


def test_centroid_roundtrip_and_legacy_format():
    total = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # norm 5
    back = cluster.centroid_from_blob(cluster.centroid_to_blob(total), 7)
    assert np.allclose(back / np.linalg.norm(back), total / 5, atol=1e-3) and np.isclose(np.linalg.norm(back), 7, atol=0.05)
    legacy = total.astype(np.float32).tobytes()  # old rows: plain float32 sums
    assert np.array_equal(cluster.centroid_from_blob(legacy, 7), total)


def test_active_seconds_estimate():
    # statements 1 min apart for 10 min, then a long gap, then a final statement: tail counted per burst end
    stamps = [0, 60, 120, 1000, 1060]
    assert db.estimated_active_seconds(stamps) == 60 + 60 + 300 + 60 + db.SUSPEND_AFTER_S
