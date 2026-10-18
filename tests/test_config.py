import os

from nazariya.config import load_dotenv, load_feeds, load_outlets, runner_of
from nazariya.ingest import select_outlets

HOME = {"indianexpress", "news18", "firstpost", "jansatta", "organiser"}


def test_runner_partition_covers_every_outlet_once():
    outlets, feeds = load_outlets(), load_feeds()
    cloud = {o["id"] for o in select_outlets(outlets, feeds, None, False, "cloud")}
    home = {o["id"] for o in select_outlets(outlets, feeds, None, False, "home")}
    everything = {o["id"] for o in select_outlets(outlets, feeds, None, False, "all")}
    assert home == HOME
    assert not cloud & home
    assert cloud | home == everything == {o["id"] for o in outlets}


def test_runner_default_is_cloud():
    assert runner_of({"id": "x"}) == "cloud"


def test_load_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nNAZ_A="from-file"\nNAZ_B=from-file\n', encoding="utf-8")
    monkeypatch.delenv("NAZ_A", raising=False)
    monkeypatch.setenv("NAZ_B", "from-env")
    load_dotenv(env)
    assert os.environ["NAZ_A"] == "from-file"
    assert os.environ["NAZ_B"] == "from-env"
