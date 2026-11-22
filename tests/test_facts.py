from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import insert, select

from nazariya import facts, inbox
from nazariya.analytics.attention import _km, _logrank
from nazariya.analytics.outlets import _ca, wilson
from nazariya.db import (
    article_events, article_text, article_vectors, articles, copies, coverage, entities, event_facts, events,
    get_engine, mentions, outlets,
)
from nazariya.ingest import headline_edits

NOW = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)


def test_death_toll():
    assert facts.death_toll("Bus falls into gorge, 12 dead in Himachal") == 12
    assert facts.death_toll("Death toll rises to 1,203 after quake") == 1203
    assert facts.death_toll("हादसे में 7 लोगों की मौत") == 7
    assert facts.death_toll("Sensex falls 300 points") is None
    assert facts.death_toll("") is None


def test_headline_edits_ignores_slugs_and_cosmetic_changes():
    known = {
        "https://x/a": (1, "Minister resigns over scam", NOW),
        "https://x/b": (2, "Minister's  ‘big’ claim", NOW),
        "https://x/teenager-held-on-charge/1": (3, "Teenager held on charge", NOW),  # slug title
    }
    feed = {"https://x/a": "Minister quits amid probe", "https://x/b": "minister's 'big' claim",
            "https://x/teenager-held-on-charge/1": "Teen held", "https://x/new": "New story"}
    out = headline_edits(known, feed, "o", NOW)
    assert [(e["article_id"], e["old_title"], e["new_title"]) for e in out] == \
        [(1, "Minister resigns over scam", "Minister quits amid probe")]


def _seed(engine):
    with engine.begin() as conn:
        conn.execute(insert(outlets), [
            {"id": "en1", "name": "En One", "lang": "en", "region": "national", "role": "outlet"},
            {"id": "en2", "name": "En Two", "lang": "en", "region": "national", "role": "outlet"},
            {"id": "hi1", "name": "Hi One", "lang": "hi", "region": "national", "role": "outlet"},
            {"id": "pib", "name": "PIB", "lang": "en", "region": "national", "role": "reference"},
        ])
        conn.execute(insert(entities), [
            {"qid": "Q9357528", "type": "state", "label_en": "Delhi"},
            {"qid": "Q1193", "type": "state", "label_en": "Manipur"},
            {"qid": "Q1", "type": "person", "label_en": "A Politician"},
        ])
        t = NOW - timedelta(hours=10)
        conn.execute(insert(events).values(id=1, first_seen=t, last_seen=t + timedelta(minutes=90), n_articles=4,
                                           n_outlets=4, title="Manipur floods", model="m", updated_at=NOW))
        rows = [  # id, outlet, minutes after t, lang, agency, title
            (1, "pib", 0, "en", None, "Relief for Manipur flood victims"),
            (2, "en1", 30, "en", None, "Manipur floods: 14 dead, minister visits"),
            (3, "en2", 90, "en", "PTI", "Manipur floods kill 14"),
            (4, "hi1", 60, "hi", None, "मणिपुर में बाढ़, 14 लोगों की मौत"),
        ]
        for aid, o, m, lang, ag, title in rows:
            conn.execute(insert(articles).values(
                id=aid, outlet_id=o, url=f"https://x/{aid}", title=title, lang=lang, published_at=t + timedelta(minutes=m),
                date_fixed=False, discovered_at=t, status="extracted", attempts=1, agency=ag,
                syndicated_of=2 if aid == 3 else None))
            conn.execute(insert(article_events).values(article_id=aid, event_id=1, method="online", assigned_at=NOW))
        conn.execute(insert(article_text), [
            {"article_id": 1, "text": "The government announced relief.", "expires_at": NOW + timedelta(days=1)},
            {"article_id": 2, "text": "NEW DELHI: Floods in Manipur killed 14.", "expires_at": NOW + timedelta(days=1)},
        ])
        conn.execute(insert(mentions), [
            {"article_id": 1, "qid": "Q1193", "n": 2, "in_title": True},
            {"article_id": 2, "qid": "Q1193", "n": 2, "in_title": True},
            {"article_id": 2, "qid": "Q9357528", "n": 1, "in_title": False},  # the dateline only
            {"article_id": 2, "qid": "Q1", "n": 1, "in_title": False},
        ])
        v_en = np.array([1, 0, 0, 0], dtype=np.float16).tobytes()
        v_hi = np.array([0.6, 0.8, 0, 0], dtype=np.float16).tobytes()
        conn.execute(insert(article_vectors), [
            {"article_id": 2, "model": "m", "vec": v_en, "text_kind": "lede", "created_at": NOW},
            {"article_id": 3, "model": "m", "vec": v_en, "text_kind": "lede", "created_at": NOW},
            {"article_id": 4, "model": "m", "vec": v_hi, "text_kind": "lede", "created_at": NOW},
        ])


def test_facts_summarise_an_event(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    _seed(engine)
    out = facts.run(engine, now=NOW)
    assert out["events"] == 1 and out["coverage_rows"] == 4 and out["copies"] == 1
    with engine.connect() as conn:
        f = conn.execute(select(event_facts)).mappings().one()
        cov = {r.outlet_id: r for r in conn.execute(select(coverage))}
        cp = conn.execute(select(copies)).mappings().one()
    assert f["first_outlet"] == "pib" and f["govt"] and f["wire"] and f["toll"] == 14
    assert f["states"] == ["Q1193"]                   # Delhi was only the dateline
    assert f["delhi_dateline"] and f["n_politicians"] == 1
    assert f["first_report"]["govt"] and f["first_report"]["states"] == ["Q1193"] and f["first_report"]["toll"] is None
    assert f["langs"] == {"en": 3, "hi": 1}
    assert abs(f["lang_drift"] - 0.4) < 0.01 and f["lang_drift_n"] == 3
    assert cov["en2"].n_wire == 1 and cov["en2"].first_own_at is None  # PTI copy is not own reporting
    assert cov["en1"].first_own_at is not None
    assert cov["hi1"].first_url == "https://x/4" and cov["hi1"].first_article_id == 4
    assert cp["outlet_id"] == "en2" and cp["source_outlet_id"] == "en1" and cp["agency"] == "PTI"
    assert facts.run(engine, now=NOW)["events"] == 1  # idempotent


def test_inbox_roundtrip(tmp_path):
    home = get_engine(f"sqlite:///{tmp_path / 'home.db'}")
    cloud = get_engine(f"sqlite:///{tmp_path / 'cloud.db'}")
    _seed(home)
    path = tmp_path / "inbox.jsonl.gz"
    assert inbox.export(home, path, days=3, now=NOW) == 4
    with cloud.begin() as conn:  # cloud already has an article with a different id
        conn.execute(insert(outlets).values(id="en1", name="En One", lang="en", role="outlet"))
        conn.execute(insert(articles).values(id=50, outlet_id="en1", url="https://x/other", title="Other",
                                             discovered_at=NOW, status="pending", attempts=0, date_fixed=False))
    assert inbox.import_(cloud, path, now=NOW) == {"articles": 4, "texts": 2, "vectors": 3}
    with cloud.connect() as conn:
        assert conn.execute(select(articles.c.id).where(articles.c.url == "https://x/2")).scalar() > 50
        assert len(conn.execute(select(article_vectors)).all()) == 3
    inbox.import_(cloud, path, now=NOW)  # importing twice changes nothing
    with cloud.connect() as conn:
        assert len(conn.execute(select(articles)).all()) == 5


def test_kaplan_meier_and_logrank():
    curve, med = _km([1, 2, 3, 4], [True, True, False, True], [0, 1, 2, 5])
    assert curve == [1.0, 0.75, 0.5, 0.0] and med == 2
    assert _logrank(([1, 1, 1], [True] * 3), ([9, 9, 9], [True] * 3)) > 3


def test_correspondence_analysis_separates_blocks():
    X = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0]], dtype=float)
    rows, _, share = _ca(X, 2)
    assert np.sign(rows[0, 0]) == np.sign(rows[1, 0]) != np.sign(rows[2, 0])
    assert 0 < share[0] <= 1
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi


def test_missing_columns_are_added(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)  # a coverage table from before the link columns existed
    c.execute("CREATE TABLE coverage (event_id INTEGER, outlet_id VARCHAR(40), first_at DATETIME NOT NULL, "
              "first_own_at DATETIME, n_articles INTEGER NOT NULL, n_wire INTEGER NOT NULL, PRIMARY KEY (event_id, outlet_id))")
    c.commit(); c.close()
    get_engine(f"sqlite:///{path}")
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(coverage)")}
    assert {"n_text", "first_article_id", "first_url", "first_title"} <= cols


def test_weekly_edition_freezes_once_on_monday(tmp_path, monkeypatch):
    from nazariya.analytics import pages
    monkeypatch.setattr(pages, "DATA", tmp_path / "site")
    engine = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    monday = datetime(2026, 11, 30, 2, tzinfo=timezone.utc)  # 07:30 IST, a Monday
    out = pages.editions_step(engine, {"events": 5}, monday)
    assert out == {"editions": 1, "new": "2026-W48"}  # the week that just ended
    assert pages.editions_step(engine, {"events": 5}, monday + timedelta(hours=4))["new"] is None
    assert pages.editions_step(engine, {"events": 5}, monday + timedelta(days=1))["new"] is None  # Tuesday
    import json
    idx = json.loads((tmp_path / "site" / "editions" / "index.json").read_text(encoding="utf-8"))
    assert idx[0]["issue"] == "2026-W48" and idx[0]["number"] == 1
    body = json.loads((tmp_path / "site" / "editions" / "2026-W48.json").read_text(encoding="utf-8"))
    assert body["issue"]["number"] == 1


def test_positions_are_rotated_onto_last_week(tmp_path):
    from nazariya.analytics import pages
    from nazariya.analytics.data import Facts
    from nazariya.db import outlet_positions
    engine = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    base = {"a": (1, 0), "b": (0, 1), "c": (-1, 0), "d": (0, -1), "e": (0.5, 0.5)}
    def result(rot):
        c, s = np.cos(rot), np.sin(rot)
        return {"outlet_map": {"ready": True, "points": [
            {"id": o, "x": x * c - y * s, "y": x * s + y * c, "events": 10} for o, (x, y) in base.items()]}}
    week1 = Facts(datetime(2026, 11, 20, tzinfo=timezone.utc), NOW, {}, {}, {})
    week2 = Facts(datetime(2026, 11, 27, tzinfo=timezone.utc), NOW, {}, {}, {})
    pages.record_positions(engine, week1, result(0))
    pages.record_positions(engine, week2, result(np.pi / 2))  # same map, rotated: should land back
    with engine.connect() as conn:
        w2 = {r.outlet_id: (r.x, r.y) for r in conn.execute(select(outlet_positions).where(outlet_positions.c.week == "2026-W48"))}
    for o, (x, y) in base.items():
        assert abs(w2[o][0] - x) < 1e-6 and abs(w2[o][1] - y) < 1e-6
