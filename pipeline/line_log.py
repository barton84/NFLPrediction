"""Line movement log and closing line value (CLV).

site/data/line_log.json is a permanent record, committed to the repo by the
daily workflow. For every game it keeps one snapshot per day until kickoff:
the posted line, where it came from (DraftKings or consensus), the model's
number, and the model version that produced it. Snapshots are never
overwritten by later days, so the line from when the model first posted is
always there.

Closing line value compares the first snapshot (when the model posted) with
the last snapshot before kickoff (game-day morning). When both came from
DraftKings that is a same-book comparison. If no snapshot exists from the
game day, the consensus closing line from nflverse is used instead.

"Move toward model" = how many points the market moved in the direction of
the model's side. Positive means the market came around to the model.
"""
import json, os, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")


def kickoff_et(date, time):
    h, m = (time or "13:00").split(":")
    d = datetime.date.fromisoformat(date)
    return datetime.datetime(d.year, d.month, d.day, int(h), int(m), tzinfo=ET)


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def record(log, games, version, now=None):
    """Add today's snapshot for every game that has not kicked off yet."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    today = now.astimezone(CT).date().isoformat()
    for g in games:
        if g.get("line") is None or g.get("result") is not None:
            continue
        if kickoff_et(g["date"], g["time"]) <= now:
            continue
        e = log.setdefault(g["id"], {"away": g["away"], "home": g["home"], "date": g["date"], "time": g["time"],
                                     "season": g.get("season"), "week": g.get("week"), "snaps": []})
        snap = {"day": today, "line": g["line"], "src": g["line_src"], "model": g["model"], "version": version}
        if e["snaps"] and e["snaps"][-1]["day"] == today:
            e["snaps"][-1] = snap          # re-run on the same day: keep the latest
        else:
            e["snaps"].append(snap)
    return log


def settle(log, finals):
    """finals: game_id -> (consensus closing line, home margin). Adds results once games are played."""
    for gid, e in log.items():
        if gid in finals and e.get("result") is None:
            close, res = finals[gid]
            e["close_consensus"], e["result"] = close, res
    return log


def clv_rows(log):
    rows = []
    for gid, e in log.items():
        if e.get("result") is None or not e.get("snaps"):
            continue
        first = e["snaps"][0]
        last = e["snaps"][-1]
        if last["day"] == e["date"] and len(e["snaps"]) > 1:
            close, close_src = last["line"], last["src"] + " game-day"
        elif e.get("close_consensus") is not None:
            close, close_src = e["close_consensus"], "Consensus close"
        else:
            close, close_src = last["line"], last["src"] + " last seen"
        edge = first["model"] - first["line"]
        if abs(edge) < 1e-9:
            side_home, move = None, 0.0
        else:
            side_home = edge > 0
            move = (close - first["line"]) if side_home else (first["line"] - close)
        rows.append({"id": gid, "away": e["away"], "home": e["home"], "week": e.get("week"), "season": e.get("season"),
                     "posted_day": first["day"], "posted_line": first["line"], "posted_src": first["src"],
                     "model": first["model"], "edge": round(edge, 2), "side_home": side_home,
                     "close": close, "close_src": close_src, "move": round(move, 2),
                     "version": first.get("version"), "result": e["result"]})
    return rows


def summarize(rows):
    def agg(rs):
        if not rs:
            return {"n": 0}
        return {"n": len(rs), "toward": sum(r["move"] > 0 for r in rs), "away": sum(r["move"] < 0 for r in rs),
                "flat": sum(r["move"] == 0 for r in rs), "avg": round(sum(r["move"] for r in rs) / len(rs), 3)}
    weeks = sorted({(r["season"], r["week"]) for r in rows})
    by_week = [{"season": s, "week": w, **agg([r for r in rows if (r["season"], r["week"]) == (s, w)])} for s, w in weeks]
    buckets = [("Under 1 pt", 0, 1), ("1 to 3 pts", 1, 3), ("3 to 5 pts", 3, 5), ("5+ pts", 5, 99)]
    by_edge = [{"bucket": b, **agg([r for r in rows if lo <= abs(r["edge"]) < hi])} for b, lo, hi in buckets]
    versions = sorted({r["version"] for r in rows if r.get("version")})
    by_version = [{"version": v, **agg([r for r in rows if r.get("version") == v])} for v in versions]
    return {"season": agg(rows), "by_week": by_week, "by_edge": by_edge, "by_version": by_version}
