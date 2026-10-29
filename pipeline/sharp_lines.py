"""Circa and Pinnacle line history from OddsPapi (free plan: 250 requests a month).

OddsPapi's historical-odds endpoint returns every price change for a game, so
we do not need to poll. Each game is pulled twice:
  * "early": the first line check on Wednesday (opener and early moves)
  * "final": the first line check within 100 minutes of kickoff (or just after
    it, if GitHub ran late); snapshots after kickoff are dropped
One call per game covers both books (the endpoint takes up to 3). With the
fixtures lookup that is roughly 165 requests a month.

Output: site/data/sharp_lines.json
  {game_id: {"pulled": {"early": ts, "final": ts},
             "books": {"circa": [{"t", "line", "ph", "pa"}], "pin": [...]}}}
"line" uses the site convention (positive = home favored). Each list holds
only the points where the main line or a price changed.

Needs env var ODDSPAPI_KEY (GitHub secret). Without it this does nothing.
Run with --probe to print the raw shape of the API responses for one game.
"""
import os, sys, json, time, datetime, urllib.request, urllib.parse
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(__file__))
from fetch_odds import NAME_TO_ABBR
import line_log

API = "https://api.oddspapi.io/v4"
SPORT = 14                                    # American football
BOOKS = {"circasports": "circa", "pinnacle": "pin"}
OUT = os.path.join(os.path.dirname(__file__), "..", "site", "data")
CT = ZoneInfo("America/Chicago")
PROBE = "--probe" in sys.argv


def api(path, **params):
    params["apiKey"] = os.environ["ODDSPAPI_KEY"]
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
            time.sleep(1.1)                       # their docs ask for ~1 second between calls
            return data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            if e.code == 429 and attempt < 3:
                try:
                    wait = json.loads(body).get("retryMs", 2000) / 1000
                except Exception:
                    wait = 2
                time.sleep(wait + 0.5)
                continue
            print(f"oddspapi {path} failed: HTTP {e.code} {body}")
            return None
    return None


def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        for k in ("data", "fixtures", "markets", "results"):
            if isinstance(x.get(k), list):
                return x[k]
    return []


def american(dec):
    try:
        d = float(dec)
    except (TypeError, ValueError):
        return None
    if d <= 1.0:
        return None
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def parse_ts(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return datetime.datetime.fromtimestamp(s / 1000 if s > 1e11 else s, datetime.timezone.utc)
    return datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


# ---------- market catalog: which market ids are spreads, and their handicap ----------
def spread_catalog():
    """marketId -> (handicap, [outcomeId, outcomeId]) for full-game spread markets.
    Cached in site/data/oddspapi_markets.json so it costs one request ever."""
    path = os.path.join(OUT, "oddspapi_markets.json")
    if os.path.exists(path):
        cat = json.load(open(path))
    else:
        raw = api("markets", sportId=SPORT)
        if raw is None:
            return {}
        cat = {}
        for m in as_list(raw):
            name = (m.get("marketName") or "").lower()
            if not name.startswith("handicap") or "half" in name or "quarter" in name or m.get("handicap") is None:
                continue
            if "period" in m and m.get("period") not in (None, "", "result", "fulltime", "ft"):
                continue
            outs = [str(o.get("outcomeId")) for o in m.get("outcomes", []) if o.get("outcomeId") is not None]
            cat[str(m["marketId"])] = [float(m["handicap"]), sorted(outs, key=lambda v: int(v) if v.isdigit() else v)]
        if cat:
            json.dump(cat, open(path, "w"), indent=0, sort_keys=True)
        print("oddspapi spread markets in catalog:", len(cat))
    return cat


# ---------- turn one book's history into a main-line timeline ----------
def timeline(book_hist, cat, kickoff):
    """book_hist: {marketId: {"outcomes": {outcomeId: {"players": {"0": [snapshots]}}}}}
    Returns [(t, handicap_on_outcome1, dec_price_outcome1, dec_price_outcome2)] at each change,
    picking the most balanced spread market (the book's main line) at every moment."""
    events = []        # (t, marketId, outcome index 0/1, decimal price or None if pulled)
    for mid, m in (book_hist.get("markets") or {}).items():
        if str(mid) not in cat:
            continue
        hcap, outs = cat[str(mid)]
        for oid, o in (m.get("outcomes") or {}).items():
            if str(oid) not in outs:
                continue
            idx = outs.index(str(oid))
            snaps = ((o.get("players") or {}).get("0")) or []
            if isinstance(snaps, dict):
                snaps = [snaps]
            for s in snaps:
                t = parse_ts(s.get("createdAt") or s.get("changedAt") or s.get("timestamp"))
                if t is None or t >= kickoff:
                    continue
                price = s.get("price")
                if s.get("active") is False:
                    price = None
                events.append((t, str(mid), idx, price))
    events.sort(key=lambda e: e[0])
    cur, seen, out = {}, {}, []
    for t, mid, idx, price in events:
        cur[(mid, idx)] = price
        seen[mid] = t
        # main line = the most balanced market; near-ties go to the one the book touched last
        cands = []
        for m_ in {k[0] for k in cur}:
            p1, p2 = cur.get((m_, 0)), cur.get((m_, 1))
            if p1 and p2:
                cands.append((abs(float(p1) - float(p2)), m_))
        if not cands:
            continue
        low = min(c[0] for c in cands)
        best = max((m_ for g_, m_ in cands if g_ <= low + 0.05), key=lambda m_: seen[m_])
        if best is None:
            continue
        pt = (t, cat[best][0], float(cur[(best, 0)]), float(cur[(best, 1)]))
        if out and out[-1][1:] == pt[1:]:
            continue
        if out and (t - out[-1][0]).total_seconds() < 60:
            out[-1] = pt                      # same minute: keep the latest
        else:
            out.append(pt)
    return out


def orient(tl, dk_line, home_is_p1):
    """Convert to site convention. Outcome 1 is assumed to be participant 1. If that does
    not match DraftKings at all, flip it (guards against a different outcome order)."""
    def conv(flip):
        res = []
        for t, h, p1, p2 in tl:
            home_first = home_is_p1 != flip
            home_h = h if home_first else -h
            ph, pa = (p1, p2) if home_first else (p2, p1)
            res.append({"t": t.strftime("%Y-%m-%dT%H:%MZ"), "line": -home_h, "ph": american(ph), "pa": american(pa)})
        return res
    a, b = conv(False), conv(True)
    if dk_line is None or not a:
        return a
    return a if abs(a[-1]["line"] - dk_line) <= abs(b[-1]["line"] - dk_line) else b


# ---------- main ----------
def due(entry, kickoff, now):
    """Which pull this game needs now, if any."""
    pulled = entry.get("pulled", {})
    if "final" not in pulled and kickoff - datetime.timedelta(minutes=100) <= now <= kickoff + datetime.timedelta(hours=6):
        return "final"
    if "early" not in pulled and "final" not in pulled and now.astimezone(CT).weekday() == 2 and now < kickoff:
        return "early"
    return None


def main(now=None):
    if not os.environ.get("ODDSPAPI_KEY"):
        print("ODDSPAPI_KEY not set; skipping Circa / Pinnacle")
        return
    now = now or datetime.datetime.now(datetime.timezone.utc)
    D = json.load(open(os.path.join(OUT, "model.json")))
    path = os.path.join(OUT, "sharp_lines.json")
    store = json.load(open(path)) if os.path.exists(path) else {}
    todo = []
    for g in D.get("this_week", []):
        kick = line_log.kickoff_et(g["date"], g["time"])
        kind = "final" if PROBE else due(store.get(g["id"], {}), kick, now)
        if kind:
            todo.append((g, kick, kind))
    if PROBE:
        todo = todo[:1]
    if not todo:
        print("Circa / Pinnacle: nothing due this run")
        return
    # find OddsPapi fixture ids for the games we need (one request covers up to 10 days)
    dates = sorted(g["date"] for g, _, _ in todo)
    d0 = datetime.date.fromisoformat(dates[0]) - datetime.timedelta(days=1)
    d1 = min(datetime.date.fromisoformat(dates[-1]) + datetime.timedelta(days=1), d0 + datetime.timedelta(days=9))
    fx = as_list(api("fixtures", sportId=SPORT, **{"from": d0.isoformat(), "to": d1.isoformat()}))
    if PROBE:
        print("PROBE fixture sample:", json.dumps(fx[:1], default=str)[:1500])
    fixtures = {}
    for f in fx:
        if "nfl" not in (f.get("tournamentName") or "").lower():
            continue
        p1, p2 = NAME_TO_ABBR.get(f.get("participant1Name")), NAME_TO_ABBR.get(f.get("participant2Name"))
        if p1 and p2:
            fixtures[frozenset((p1, p2))] = (f, p1)
    cat = spread_catalog()
    if not cat:
        print("no spread markets found in the OddsPapi catalog; stopping")
        return
    n_ok = 0
    for g, kick, kind in todo:
        hit = fixtures.get(frozenset((g["home"], g["away"])))
        if not hit:
            print(f"no OddsPapi fixture for {g['away']}@{g['home']}")
            if not PROBE:   # mark it tried, so a missing game never eats requests run after run
                store.setdefault(g["id"], {"pulled": {}, "books": {}})["pulled"][kind] = "no fixture"
            continue
        f, p1 = hit
        raw = api("historical-odds", fixtureId=f["fixtureId"], bookmakers=",".join(BOOKS))
        if raw is None:
            continue
        if PROBE:
            print("PROBE historical keys:", list(raw.keys()) if isinstance(raw, dict) else type(raw))
            print("PROBE historical sample:", json.dumps(raw, default=str)[:3000])
        books_raw = raw.get("bookmakers", raw) if isinstance(raw, dict) else {}
        entry = store.setdefault(g["id"], {"pulled": {}, "books": {}})
        for slug, code in BOOKS.items():
            tl = timeline(books_raw.get(slug) or {}, cat, kick)
            pts = orient(tl, g.get("line"), home_is_p1=(p1 == g["home"]))
            if pts:
                entry["books"][code] = pts
            print(f"{g['away']}@{g['home']} {code}: {len(pts)} points" + (f", latest {pts[-1]}" if pts else ""))
        entry["pulled"][kind] = now.strftime("%Y-%m-%dT%H:%MZ")
        n_ok += 1
    if not PROBE:
        with open(path, "w") as fh:
            json.dump(store, fh, separators=(",", ":"), sort_keys=True)
    print(f"Circa / Pinnacle: pulled {n_ok} of {len(todo)} games due")


if __name__ == "__main__":
    main()
