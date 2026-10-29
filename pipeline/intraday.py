"""Daytime line checks: DraftKings spreads only, no model rebuild.

Runs 8 times a day from .github/workflows/line_checks.yml with the same Odds API
key as the morning run (ODDS_API_KEY). Each run costs 1 credit. It:
  1. pulls current DraftKings, FanDuel and Caesars spreads (still 1 credit),
  2. adds a timestamped "check" snapshot to site/data/line_log.json for every
     game the morning run is already tracking and that has not kicked off,
  3. updates that game in site/data/model.json: line, prices, edge, cover
     chance and line history, so the site shows the new line right away.
The model's numbers, ratings and totals only change on the morning run.
Needs only numpy (for the cover chance).
"""
import os, sys, json, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import fetch_odds, line_log
import probability as P

OUT = os.path.join(os.path.dirname(__file__), "..", "site", "data")


def main(dk=None, now=None):
    if dk is None:
        dk = fetch_odds.fetch(markets="spreads")
    if not dk:
        print("no lines pulled; nothing to do")
        return
    log_path, model_path = os.path.join(OUT, "line_log.json"), os.path.join(OUT, "model.json")
    log = line_log.load(log_path)
    D = json.load(open(model_path))
    kw = np.array(D["prob"]["kw"]); K = D["prob"].get("K") or 0
    now = now or datetime.datetime.now(datetime.timezone.utc)
    n_log = n_moved = 0
    for g in D.get("this_week", []):
        rec = dk.get(f"{g['away']}@{g['home']}")
        if not rec or rec.get("home_spread") is None:
            continue
        if abs((datetime.date.fromisoformat(rec["commence"][:10]) - datetime.date.fromisoformat(g["date"])).days) > 1:
            continue   # a different meeting of the same teams
        line = -rec["home_spread"]   # site convention: positive = home favored
        books = {}
        for c in line_log.BOOK_CODES:
            b = rec.get(c) or {}
            books[f"{c}_line"] = -b["home_spread"] if b.get("home_spread") is not None else None
            books[f"{c}_price_home"], books[f"{c}_price_away"] = b.get("home_price"), b.get("away_price")
        if not line_log.add_check(log, g["id"], line, rec.get("home_price"), rec.get("away_price"), now, books=books):
            continue
        n_log += 1
        n_moved += g.get("line") != line
        g.update(line=line, line_src="DK", price_home=rec.get("home_price"), price_away=rec.get("away_price"),
                 edge=round(g["model"] - line, 2), **books)
        c = P.market_center(line, kw) + K * (g["model"] - line)
        a, pp = P.home_cover(line, c, kw)
        home_side = g["model"] > line
        g.update(cover=round(a if home_side else 1 - a - pp, 3), push=round(pp, 3), side_home=bool(home_side))
        g["history"] = [{k: s.get(k) for k in line_log.SNAP_KEYS} for s in log[g["id"]]["snaps"]]
    D["meta"]["lines_checked"] = now.isoformat()
    D["meta"]["dk_lines_this_week"] = sum(1 for x in D.get("this_week", []) if x.get("line_src") == "DK")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1, sort_keys=True)
    with open(model_path, "w") as f:
        json.dump(D, f, separators=(",", ":"), allow_nan=False)
    print(f"line check {now:%Y-%m-%d %H:%M}Z: {n_log} games logged, {n_moved} lines moved")


if __name__ == "__main__":
    main()
