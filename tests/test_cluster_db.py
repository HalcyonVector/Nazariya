from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import insert, select

from nazariya import cluster
from nazariya.db import article_events, article_vectors, articles, events, get_engine, outlets

NOW = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def add(conn, aid, outlet, vec, hours_ago, title):
    conn.execute(insert(articles).values(
        id=aid, outlet_id=outlet, url=f"https://x/{aid}", title=title, discovered_at=NOW,
        published_at=NOW - timedelta(hours=hours_ago), status="extracted", attempts=1, date_fixed=False))
    conn.execute(insert(article_vectors).values(
        article_id=aid, model="m", vec=cluster.to_blob(vec), text_kind="lede", created_at=NOW))


def test_assign_new_creates_then_extends_events(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    with engine.begin() as conn:
        conn.execute(insert(outlets), [{"id": o, "name": o} for o in ("a", "b", "c")])
        add(conn, 1, "a", unit(1, 0, 0), 10, "Stampede in Kolkata")
        add(conn, 2, "b", unit(0.97, 0.1, 0), 9, "Kolkata stampede: toll rises")
        add(conn, 3, "c", unit(0, 1, 0), 8, "Budget session begins")

    first = cluster.assign_new(engine, NOW, tau=0.8, model="m")
    assert first == {"assigned": 3, "new_events": 2, "joined_existing": 0, "events_touched": 2}

    with engine.begin() as conn:
        add(conn, 4, "c", unit(0.95, 0.05, 0.05), 2, "Kolkata stampede: arrests")
    second = cluster.assign_new(engine, NOW, tau=0.8, model="m")
    assert second == {"assigned": 1, "new_events": 0, "joined_existing": 1, "events_touched": 1}

    with engine.connect() as conn:
        evs = {r.n_articles: r for r in conn.execute(select(events)).all()}
        members = conn.execute(select(article_events.c.event_id, article_events.c.article_id)).all()
    big = evs[3]
    assert big.n_outlets == 3
    assert big.title in {"Stampede in Kolkata", "Kolkata stampede: toll rises", "Kolkata stampede: arrests"}
    assert {aid for eid, aid in members if eid == big.id} == {1, 2, 4}
    assert evs[1].title == "Budget session begins"
