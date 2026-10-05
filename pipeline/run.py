"""Full weekly build: data -> ratings -> backtest -> site/data/model.json

Usage:  python pipeline/run.py            (downloads fresh data first)
        python pipeline/run.py --no-fetch (reuse files already in raw/)
"""
import os, sys, json, datetime, glob
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..")
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "site", "data")

import fetch_data, build_games, fetch_odds
from ratings import build
from qb import add_qb_features, load_dropbacks, REPLACEMENT, N0, DECAY
from backtest import prep, loso, report, ats

CARRY, K = 0.5, 8          # prior carryover from last season, prior weight in games
FEATURES = ["home", "d_margin", "d_sr", "d_qb_delta", "rest_diff"]
TEAM_NAMES = {v: k for k, v in fetch_odds.NAME_TO_ABBR.items()}


def nz(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


class NoIntercept(LinearRegression):
    def __init__(self):
        super().__init__(fit_intercept=False)


def main(fetch=True):
    today = datetime.date.today()
    season = today.year if today.month >= 8 else today.year - 1
    if fetch:
        fetch_data.main(season)
    build_games.main()
    g = pd.read_parquet(os.path.join(RAW, "game_stats.parquet"))
    g = add_qb_features(g)
    df, snaps = build(g, carry=CARRY, k=K)

    # ---------- which weeks are "last" and "this" ----------
    cur = df[df.season == season]
    done_frac = cur.groupby("week").apply(lambda x: x.result.notna().mean(), include_groups=False)
    last_week = int(done_frac[done_frac >= 0.5].index.max()) if (done_frac >= 0.5).any() else 0
    this_week = last_week + 1

    # ---------- backtest (leave one season out) ----------
    test_seasons = [s for s in range(2016, season) if s in df.season.unique()]
    import backtest
    backtest.TEST_SEASONS = test_seasons
    d = prep(df[df.season.isin(test_seasons)])
    variants = {
        "Scoring margin only": ["home", "d_margin"],
        "Margin + success rate": ["home", "d_margin", "d_sr"],
        "Margin + EPA/play": ["home", "d_margin", "d_epa"],
        "Margin + success rate + QB change": ["home", "d_margin", "d_sr", "d_qb_delta"],
        "Final model (adds rest days)": FEATURES,
        "Everything (adds EPA, YPP, pass EPA, divisional)": FEATURES + ["d_epa", "d_ypp", "d_pass_epa", "div"],
    }
    bt_rows, fold_coefs = [], None
    for name, cols in variants.items():
        p, c = loso(d, cols, NoIntercept)
        bt_rows.append(report(d, p, name))
        if cols == FEATURES:
            fold_coefs = c
            d["pred"] = p
    market_rmse = float(np.sqrt(((d.spread_line - d.result) ** 2).mean()))
    by_season = []
    for s, x in d.groupby("season"):
        a, n = ats(x.pred.values, x.spread_line.values, x.result.values)
        a5, n5 = ats(x.pred.values, x.spread_line.values, x.result.values, 5)
        by_season.append({"season": int(s), "games": int(len(x)),
                          "model_rmse": nz(np.sqrt(((x.pred - x.result) ** 2).mean())),
                          "market_rmse": nz(np.sqrt(((x.spread_line - x.result) ** 2).mean())),
                          "ats": nz(a, 3), "ats5": nz(a5, 3), "n5": n5})

    # ---------- final weights: fit on every completed backtest season ----------
    m = NoIntercept().fit(d[FEATURES], d.result)
    W = dict(zip(FEATURES, [float(v) for v in m.coef_]))
    weights = {"hfa": W["home"], "margin": W["d_margin"], "sr": W["d_sr"], "qb": W["d_qb_delta"], "rest": W["rest_diff"],
               "ranges": {f: [nz(fold_coefs[f].min(), 3), nz(fold_coefs[f].max(), 3)] for f in FEATURES}}

    def predict(rows):
        x = prep_any(rows)
        return x[FEATURES].values @ np.array([W[f] for f in FEATURES])

    def prep_any(x):
        x = x.copy()
        x["home"] = 1 - x.neutral
        x["rest_diff"] = (x.home_rest - x.away_rest).clip(-7, 7).fillna(0)
        x["d_qb_delta"] = x.d_qb_delta.fillna(0)
        return x

    # ---------- odds: DraftKings if key present, else consensus ----------
    os.makedirs(os.path.join(OUT, "odds_history"), exist_ok=True)
    dk_now = {}
    try:
        dk_now = fetch_odds.fetch()
    except Exception as e:
        print("odds fetch failed:", e)
    hist_path = os.path.join(OUT, "odds_history", f"{season}_{this_week:02d}.json")
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else {}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for k, v in dk_now.items():
        if v["commence"] > now:  # only update games that have not kicked off
            hist[k] = v
    if dk_now:
        json.dump(hist, open(hist_path, "w"), indent=1)

    def dk_for(week, away, home):
        p = os.path.join(OUT, "odds_history", f"{season}_{week:02d}.json")
        if not os.path.exists(p):
            return None
        rec = json.load(open(p)).get(f"{away}@{home}")
        return rec

    def line_info(r, week):
        dk = dk_for(week, r.away_team, r.home_team)
        if dk and dk.get("home_spread") is not None:
            return -dk["home_spread"], dk.get("total", r.total_line), "DK"
        return r.spread_line, r.total_line, "Consensus"

    # ---------- this week ----------
    tw = df[(df.season == season) & (df.week == this_week)].copy()
    tw["model"] = predict(tw)
    games = []
    for r in tw.sort_values(["gameday", "gametime"]).itertuples():
        line, total, src = line_info(r, this_week)
        proj_total = r.proj_home_pts + r.proj_away_pts
        games.append({
            "id": r.game_id, "away": r.away_team, "home": r.home_team, "date": r.gameday, "time": r.gametime,
            "weekday": r.weekday, "roof": r.roof if isinstance(r.roof, str) else None, "stadium": r.stadium,
            "neutral": int(r.neutral), "rest_home": nz(r.home_rest, 0), "rest_away": nz(r.away_rest, 0),
            "home_qb": r.home_qb_name, "away_qb": r.away_qb_name,
            "home_qb_delta": nz(r.home_qb_delta, 3) or 0, "away_qb_delta": nz(r.away_qb_delta, 3) or 0,
            "line": nz(line, 1), "total": nz(total, 1), "line_src": src,
            "model": nz(r.model, 2), "proj_total": nz(proj_total, 1),
            "edge": nz(r.model - line, 2) if line is not None and not pd.isna(line) else None,
            "result": nz(r.result, 0), "home_score": nz(r.home_score, 0), "away_score": nz(r.away_score, 0),
        })

    # ---------- last week + season to date ----------
    def graded(week):
        x = df[(df.season == season) & (df.week == week)].copy()
        if x.empty:
            return []
        x["model"] = predict(x)
        out = []
        for r in x.sort_values(["gameday", "gametime"]).itertuples():
            line, total, src = line_info(r, week)
            row = {"id": r.game_id, "away": r.away_team, "home": r.home_team, "line": nz(line, 1), "line_src": src,
                   "model": nz(r.model, 2), "home_score": nz(r.home_score, 0), "away_score": nz(r.away_score, 0),
                   "result": nz(r.result, 0), "home_qb": r.home_qb_name, "away_qb": r.away_qb_name}
            if pd.notna(r.result) and line is not None and not pd.isna(line):
                row["error"] = nz(r.model - r.result, 1)
                side, res = np.sign(r.model - line), np.sign(r.result - line)
                row["ats"] = "push" if res == 0 else ("win" if side == res else "loss")
                row["edge"] = nz(r.model - line, 2)
            out.append(row)
        return out

    season_weeks = []
    for w in range(1, last_week + 1):
        rows = graded(w)
        dec = [r for r in rows if r.get("ats") in ("win", "loss")]
        big = [r for r in dec if abs(r["edge"]) >= 5]
        errs = [abs(r["error"]) for r in rows if "error" in r]
        mkt = [abs(r["result"] - r["line"]) for r in rows if "error" in r]
        season_weeks.append({"week": w, "wins": sum(r["ats"] == "win" for r in dec), "losses": sum(r["ats"] == "loss" for r in dec),
                             "big_wins": sum(r["ats"] == "win" for r in big), "big_losses": sum(r["ats"] == "loss" for r in big),
                             "mae": nz(np.mean(errs), 1) if errs else None, "market_mae": nz(np.mean(mkt), 1) if mkt else None})

    # ---------- team ratings for the matchup tool and table ----------
    snap = snaps[(season, this_week)]
    played = df[(df.season == season) & df.result.notna()]
    db = load_dropbacks()
    teams = {}
    for t in sorted(snap["net"]["margin"]):
        h, a = played[played.home_team == t], played[played.away_team == t]
        n = len(h) + len(a)
        def avg(hc, ac):
            v = pd.concat([h[hc], a[ac]])
            return nz(v.mean(), 3) if n else None
        last = pd.concat([h.assign(qb=h.home_qb_id, qbn=h.home_qb_name), a.assign(qb=a.away_qb_id, qbn=a.away_qb_name)]).sort_values("week")
        # team's QB baseline = avg rating of starters in its last 8 games (incl. last season)
        allg = df[((df.home_team == t) | (df.away_team == t)) & df.result.notna()].sort_values(["season", "week"]).tail(8)
        starters = [r.home_qb_id if r.home_team == t else r.away_qb_id for r in allg.itertuples()]
        from qb import qb_rating_before
        base = float(np.mean([qb_rating_before(db, q, season, this_week)[0] for q in starters])) if starters else REPLACEMENT
        cur_qb = last.qbn.iloc[-1] if len(last) else None
        cur_id = last.qb.iloc[-1] if len(last) else (starters[-1] if starters else None)
        teams[t] = {
            "name": TEAM_NAMES.get(t, t), "games": n,
            "r_margin": nz(snap["net"]["margin"][t], 3), "r_sr": nz(snap["net"]["sr"][t], 4),
            "r_epa": nz(snap["net"]["epa"][t], 4), "off": nz(snap["off"][t], 2), "def": nz(snap["def"][t], 2),
            "ppg": avg("home_score", "away_score"), "papg": avg("away_score", "home_score"),
            "off_epa": avg("home_epa", "away_epa"), "def_epa": avg("away_epa", "home_epa"),
            "off_sr": avg("home_sr", "away_sr"), "def_sr": avg("away_sr", "home_sr"),
            "off_ypp": avg("home_ypp", "away_ypp"), "def_ypp": avg("away_ypp", "home_ypp"),
            "qb": cur_qb, "qb_id": cur_id, "qb_rating": nz(qb_rating_before(db, cur_id, season, this_week)[0], 3) if cur_id else None,
            "qb_base": nz(base, 3),
        }

    # QB list for the injury panel (anyone with 100+ dropbacks since two seasons ago)
    recent = db[db.season >= season - 2]
    ids = recent.groupby("passer_player_id").n.sum()
    ids = ids[ids >= 100].index
    names = pd.concat([g[["home_qb_id", "home_qb_name"]].set_axis(["id", "name"], axis=1),
                       g[["away_qb_id", "away_qb_name"]].set_axis(["id", "name"], axis=1)]).dropna().drop_duplicates("id", keep="last")
    names = dict(zip(names.id, names.name))
    qbs = []
    for q in ids:
        rt, nn = qb_rating_before(db, q, season, this_week)
        if q in names:
            qbs.append({"id": q, "name": names[q], "rating": nz(rt, 3), "dropbacks": nn})
    qbs.sort(key=lambda x: -x["rating"])

    out = {
        "meta": {"season": season, "this_week": this_week, "last_week": last_week, "built": now,
                 "games_in_backtest": int(len(d)), "backtest_seasons": [min(test_seasons), max(test_seasons)],
                 "carry": CARRY, "prior_games": K, "replacement_qb": REPLACEMENT,
                 "mu": nz(snap["mu"], 3), "dk_lines_this_week": sum(1 for x in games if x["line_src"] == "DK"), "hfa_note": snap["hfa"]["margin"]},
        "weights": weights, "teams": teams, "qbs": qbs, "this_week": games,
        "last_week": graded(last_week) if last_week else [], "season_weeks": season_weeks,
        "backtest": {"variants": bt_rows, "by_season": by_season, "market_rmse": nz(market_rmse, 2)},
        "weather": {"wind_per_mph_over_10": -0.4, "cold_per_deg_under_32": -0.2, "dome_total": 1.5,
                    "rain": -2.0, "snow": -3.0},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "model.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"), default=lambda o: None)
    print(f"wrote model.json: season {season} this_week {this_week} last_week {last_week}; weights", {k: round(v, 3) for k, v in W.items()})


if __name__ == "__main__":
    main(fetch="--no-fetch" not in sys.argv)
