"""Leave-one-season-out backtest. Train weights on 9 seasons, score the 10th,
rotate so every season 2016-2025 is the untouched holdout once.
Reports RMSE vs actual margin and ATS vs the closing line."""
import pandas as pd, numpy as np, os, sys, itertools, json
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
sys.path.insert(0, os.path.dirname(__file__))
from ratings import build

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")
TEST_SEASONS = list(range(2016, 2026))


def prep(df):
    d = df[df.result.notna() & df.spread_line.notna() & df.season.isin(TEST_SEASONS)].copy()
    d["home"] = 1 - d.neutral
    d["rest_diff"] = (d.home_rest - d.away_rest).clip(-7, 7)
    d["div"] = d.div_game
    return d


def ats(pred, line, actual, min_edge=0.0):
    edge = pred - line
    side = np.sign(edge)
    res = np.sign(actual - line)
    m = (np.abs(edge) >= min_edge) & (res != 0) & (side != 0)
    return (side[m] == res[m]).mean() if m.sum() else np.nan, int(m.sum())


def loso(d, cols, model_fn=LinearRegression):
    preds = pd.Series(index=d.index, dtype=float)
    coefs = []
    for s in TEST_SEASONS:
        tr, te = d[d.season != s], d[d.season == s]
        m = model_fn().fit(tr[cols], tr.result)
        preds[te.index] = m.predict(te[cols])
        coefs.append(dict(zip(cols, m.coef_), season=s, intercept=m.intercept_))
    return preds, pd.DataFrame(coefs)


def report(d, preds, name):
    rmse = np.sqrt(((preds - d.result) ** 2).mean())
    a0, n0 = ats(preds.values, d.spread_line.values, d.result.values)
    a3, n3 = ats(preds.values, d.spread_line.values, d.result.values, 3)
    a5, n5 = ats(preds.values, d.spread_line.values, d.result.values, 5)
    by = [ats(preds[d.season == s].values, d[d.season == s].spread_line.values, d[d.season == s].result.values)[0]
          for s in TEST_SEASONS]
    return dict(model=name, rmse=round(rmse, 3), ats_all=round(a0, 4), n=n0, ats_3=round(a3, 4), n3=n3,
                ats_5=round(a5, 4), n5=n5, ats_min_season=round(min(by), 3), ats_max_season=round(max(by), 3))


if __name__ == "__main__":
    g = pd.read_parquet(os.path.join(RAW, "game_stats.parquet"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if mode == "grid":
        rows = []
        for carry, k in itertools.product([0.3, 0.5, 0.7], [3, 5, 8, 12]):
            df, _ = build(g, carry=carry, k=k)
            d = prep(df)
            for cols in [["home", "d_margin"], ["home", "d_epa"], ["home", "d_margin", "d_epa"],
                         ["home", "d_margin", "d_epa", "d_sr"]]:
                p, _ = loso(d, cols)
                r = report(d, p, "+".join(cols[1:])); r.update(carry=carry, k=k); rows.append(r)
        out = pd.DataFrame(rows).sort_values("rmse")
        print(out.to_string()); out.to_csv(os.path.join(RAW, "grid.csv"))

if __name__ == "__main__" and mode == "qb":
    from qb import add_qb_features
    g = add_qb_features(g)
    g.to_parquet(os.path.join(RAW, "game_stats_qb.parquet"))
    rows = []
    for carry, k in [(0.3, 12), (0.5, 8)]:
        df, _ = build(g, carry=carry, k=k)
        d = prep(df)
        for cols in [["home", "d_margin", "d_epa", "d_sr"],
                     ["home", "d_margin", "d_epa", "d_sr", "d_qb_delta"],
                     ["home", "d_margin", "d_epa", "d_sr", "d_qb_delta", "rest_diff"],
                     ["home", "d_margin", "d_epa", "d_sr", "d_ypp", "d_pass_epa", "d_qb_delta", "rest_diff", "div"]]:
            p, c = loso(d, cols)
            r = report(d, p, "+".join(cols[1:])); r.update(carry=carry, k=k); rows.append(r)
            if "d_qb_delta" in cols:
                print(cols[-1], "coef range", c[cols].min().round(2).to_dict(), c[cols].max().round(2).to_dict())
    print(pd.DataFrame(rows).to_string())
