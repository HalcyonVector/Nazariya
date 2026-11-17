"""Row-level security on Postgres: enable_rls runs after create_all and only targets owned public tables.

The live check needs a disposable Postgres in NAZARIYA_TEST_PG_URL and skips without one; never point
it at the production database.
"""
import os

import pytest
from sqlalchemy import text

from nazariya import db


class FakeConn:
    def __init__(self, log):
        self.log = log

    def execute(self, stmt):
        self.log.append(str(stmt))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeEngine:
    def __init__(self, dialect):
        self.dialect = type("D", (), {"name": dialect})()
        self.log: list[str] = []

    def begin(self):
        return FakeConn(self.log)


def test_postgres_gets_the_rls_statement():
    engine = FakeEngine("postgresql")
    db.enable_rls(engine)
    assert len(engine.log) == 1
    sql = engine.log[0]
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "nspname = 'public'" in sql
    assert "pg_get_userbyid(c.relowner) = current_user" in sql
    assert "NOT c.relrowsecurity" in sql


def test_sqlite_is_left_alone(tmp_path):
    engine = FakeEngine("sqlite")
    db.enable_rls(engine)
    assert engine.log == []
    # and the real SQLite path still builds the schema
    db.get_engine(f"sqlite:///{tmp_path / 't.db'}")


def test_get_engine_enables_rls_after_create_all(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(db.metadata, "create_all", lambda engine: calls.append("create_all"))
    monkeypatch.setattr(db, "enable_rls", lambda engine: calls.append("enable_rls"))
    db.get_engine(f"sqlite:///{tmp_path / 't.db'}")
    assert calls == ["create_all", "enable_rls"]


@pytest.mark.skipif(not os.environ.get("NAZARIYA_TEST_PG_URL"), reason="needs a disposable Postgres")
def test_live_postgres():
    url = os.environ["NAZARIYA_TEST_PG_URL"].replace("postgresql://", "postgresql+psycopg://", 1)
    engine = db.get_engine(url)
    with engine.connect() as conn:
        off = conn.execute(text(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind IN ('r','p') AND NOT c.relrowsecurity "
            "AND pg_get_userbyid(c.relowner) = current_user")).scalars().all()
        assert off == []
        # the owner still reads and writes
        conn.execute(text("INSERT INTO outlets (id, name) VALUES ('rls_probe', 'x') ON CONFLICT DO NOTHING"))
        assert conn.execute(text("SELECT count(*) FROM outlets WHERE id = 'rls_probe'")).scalar_one() == 1
        conn.rollback()
    engine.dispose()
