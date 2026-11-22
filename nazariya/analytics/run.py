"""Compute every model from the fact tables and write site/data/analysis.json.

Usage:
    python -m nazariya.analytics.run [--days 28]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone

from ..config import ROOT
from ..db import get_engine
from . import attention, forecast, outlets, pages, selection
from .data import load

OUT = ROOT / "site" / "data" / "analysis.json"
WINDOW_DAYS = 28

MODELS = {
    "outlet_map": outlets.outlet_map,
    "lead_lag": outlets.lead_lag,
    "selection": selection.run,
    "attention": attention.attention,
    "half_life": attention.half_life,
    "wire": outlets.wire,
    "copying": outlets.copy_network,
    "headline_edits": outlets.headline_edits,
    "translation": selection.translation_drift,
    "releases": outlets.release_pickup,
    "clock": outlets.newsroom_clock,
    "ownership": outlets.ownership,
    "forecast": forecast.run,
    "removals": outlets.removed_stories,
}


def findings(r: dict) -> list[dict]:
    """Plain-language findings, each tied to the section that backs it. Only models that are
    ready contribute; every number comes from the model output, nothing is written by hand."""
    out = []
    sel = r.get("selection", {})
    if sel.get("ready") and sel["under_covered"]:
        e = sel["under_covered"][0]
        out.append({"section": "selection", "text": f"“{e['title']}” broke in {e['broke_in']}. The model "
                    f"expected about {e['expected']:.0f} other outlets to pick it up; {e['actual']} did."})
    att = r.get("attention", {})
    if att.get("ready"):
        big = [s for s in att["states"] if s["population"] >= 1e7 and s["ratio"] is not None]
        lo = min(big, key=lambda s: s["ratio"])
        hi = max(big, key=lambda s: s["ratio"])
        out.append({"section": "attention", "text": f"{lo['name']} got {lo['ratio']:.2f}x the coverage its population "
                    f"and distance from Delhi predict; {hi['name']} got {hi['ratio']:.1f}x."})
    hl = r.get("half_life", {})
    if hl.get("ready"):
        where = {s["name"]: s for s in hl["dimensions"].get("Where", [])}
        a, b = where.get("Northeast"), where.get("Other states")
        if a and b and a["half_life_days"] and b["half_life_days"]:
            out.append({"section": "half_life", "text": f"Stories about the Northeast stayed in the news for a median "
                        f"{a['half_life_days'] * 24:.0f} hours; stories about other states, {b['half_life_days'] * 24:.0f} hours."})
    w = r.get("wire", {})
    if w.get("ready"):
        lo = w["rows"][-1]
        out.append({"section": "wire", "text": f"{100 * (1 - lo['own']):.0f}% of {lo['name']}'s articles were agency "
                    f"copy or copied text, the most of any outlet."})
    ll = r.get("lead_lag", {})
    if ll.get("ready") and ll["rank"]:
        t = ll["rank"][0]
        out.append({"section": "lead_lag", "text": f"{t['name']} published ahead of the median outlet on "
                    f"{100 * t['score']:.0f}% of the stories it shared with others, the most of any outlet."})
    rp = r.get("releases", {})
    if rp.get("ready"):
        out.append({"section": "releases", "text": f"{100 * (1 - rp['reported_share']):.0f}% of {rp['releases']} government "
                    f"press releases were not picked up by any of the news outlets tracked."})
    ed = r.get("headline_edits", {})
    if ed.get("ready") and ed["by_outlet"]:
        t = ed["by_outlet"][0]
        out.append({"section": "headline_edits", "text": f"{t['name']} changed {t['edits']} "
                    f"headline{'s' if t['edits'] != 1 else ''} after publishing, the most of any outlet."})
    return out


def run(days: float = WINDOW_DAYS) -> dict:
    t0 = time.monotonic()
    engine = get_engine()
    f = load(engine, days)
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "window_days": days,
              "since": f.since.isoformat(), "events": len(f.events),
              "data_since": min((e["first_seen"] for e in f.events.values()), default=f.since).isoformat(),
              "multi_outlet_events": sum(e["n_news_outlets"] >= 2 for e in f.events.values()),
              "outlets": {o.id: {"name": o.name, "lang": o.lang, "group": o.group} for o in f.outlets.values()}}
    timings = {}
    for name, fn in MODELS.items():
        t = time.monotonic()
        try:
            result[name] = fn(f)
        except Exception as exc:  # one failing model must not take the others down
            traceback.print_exc()
            result[name] = {"ready": False, "reason": f"error: {exc}"}
        timings[name] = round(time.monotonic() - t, 1)
    result["findings"] = findings(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    steps = {
        "stories": lambda: pages.stories(f),
        "positions": lambda: pages.record_positions(engine, f, result),
        "dossiers": lambda: pages.dossiers(engine, f, result),
        "editions": lambda: pages.editions_step(engine, result, f.now),
    }
    for name, step in steps.items():
        t = time.monotonic()
        try:
            print(f"[pages] {name}: {step()}")
        except Exception:
            traceback.print_exc()
        timings[name] = round(time.monotonic() - t, 1)
    return {"events": len(f.events), "ready": [k for k in MODELS if result[k].get("ready")],
            "seconds": timings, "total_s": round(time.monotonic() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=WINDOW_DAYS)
    print(run(ap.parse_args().days))


if __name__ == "__main__":
    main()
