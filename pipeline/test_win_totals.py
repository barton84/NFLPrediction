"""Backtest: does starting each season from the market's win total beat starting from half of last year?
Same leave-one-season-out test as the main model, 2016-2025."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ratings import build
import backtest
from backtest import prep, loso, report, ats
from run import NoIntercept, FEATURES
from qb import add_qb_features
from fetch_odds import NAME_TO_ABBR

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")
backtest.TEST_SEASONS = list(range(2016, 2026))
g0 = pd.read_parquet(os.path.join(RAW, "game_stats.parquet"))

# win total -> expected win % -> points per game, scale calibrated on Week 1 closing lines (pre-game info only)
w = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "win_totals.csv"))
w["abbr"] = w.team.map(NAME_TO_ABBR)
w["games"] = np.where(w.season >= 2021, 17, 16)
w["wp"] = w.line / w.games
w["wp_c"] = w.wp - w.groupby("season").wp.transform("mean")
wp = dict(zip(zip(w.season, w.abbr), w.wp_c))
wk1 = g0[(g0.week == 1) & g0.spread_line.notna()].copy()
wk1["dwp"] = [wp.get((s, h), np.nan) - wp.get((s, a), np.nan) for s, h, a in zip(wk1.season, wk1.home_team, wk1.away_team)]
wk1 = wk1.dropna(subset=["dwp"])
X = np.c_[wk1.dwp, 1 - (wk1.location == "Neutral").astype(int)]
c, hfa = np.linalg.lstsq(X, wk1.spread_line, rcond=None)[0]
print(f"scale: 1.0 win% = {c:.1f} pts  (one extra win in 17 games = {c/17:.2f} pts); week-1 HFA {hfa:.2f}")
wt_prior = {k: c * v for k, v in wp.items()}

def run(name, gq, **kw):
    df, _ = build(gq, carry=0.5, k=8, **kw)
    d = prep(df)
    p, coefs = loso(d, FEATURES, NoIntercept)
    d["pred"] = p
    r = report(d, p, name)
    for lab, lo, hi in [("wk1-4", 1, 4), ("wk5-9", 5, 9), ("wk10+", 10, 30)]:
        x = d[(d.week >= lo) & (d.week <= hi)]
        r[lab] = round(float(np.sqrt(((x.pred - x.result) ** 2).mean())), 3)
    x = d[d.week <= 4]
    r["ats_wk1-4"] = round(ats(x.pred.values, x.spread_line.values, x.result.values)[0], 3)
    by = d.groupby("season").apply(lambda x: np.sqrt(((x.pred - x.result) ** 2).mean()), include_groups=False)
    r["by_season"] = by
    return r

g_base = add_qb_features(g0)
g_reset = add_qb_features(g0, season_reset=True)
rows = [run("Current model", g_base),
        run("Win-total start", g_base, wt_prior=wt_prior),
        run("Win-total start + QB reset", g_reset, wt_prior=wt_prior),
        run("50/50 blend + QB reset", g_reset, wt_prior=wt_prior, wt_blend=0.5),
        run("Win-total start, stronger (12 games) + QB reset", g_reset, wt_prior=wt_prior, k_margin=12)]
base = rows[0]["by_season"]
out = pd.DataFrame([{k: v for k, v in r.items() if k != "by_season"} for r in rows])
print(out[["model", "rmse", "wk1-4", "wk5-9", "wk10+", "ats_all", "ats_wk1-4", "ats_5", "n5"]].to_string())
for r in rows[1:]:
    better = int((r["by_season"] < base - 1e-9).sum())
    print(f"{r['model']}: lower error than current model in {better} of 10 seasons")
pd.DataFrame({r["model"]: r["by_season"] for r in rows}).round(3).to_csv(os.path.join(RAW, "wt_by_season.csv"))
