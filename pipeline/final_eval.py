import pandas as pd, numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from ratings import build
from backtest import prep, loso, report, ats, TEST_SEASONS
from sklearn.linear_model import LinearRegression
g = pd.read_parquet("raw/game_stats_qb.parquet")
df, _ = build(g, carry=0.5, k=8)
d = prep(df)
rows = []
sets = {"margin only": ["home","d_margin"], "margin+sr": ["home","d_margin","d_sr"], "margin+epa": ["home","d_margin","d_epa"],
        "margin+sr+qb": ["home","d_margin","d_sr","d_qb_delta"], "margin+sr+qb+rest": ["home","d_margin","d_sr","d_qb_delta","rest_diff"],
        "margin+epa+sr+qb+rest": ["home","d_margin","d_epa","d_sr","d_qb_delta","rest_diff"]}
for name, cols in sets.items():
    p, c = loso(d, cols); r = report(d, p, name); rows.append(r)
    if name == "margin+sr+qb+rest":
        best_p, best_c = p, c
print(pd.DataFrame(rows).to_string())
print(best_c.round(3).to_string())
d["pred"] = best_p
d["wk"] = pd.cut(d.week, [0,4,9,14,18,30], labels=["1-4","5-9","10-14","15-18","playoffs"])
for w, x in d.groupby("wk", observed=True):
    print(w, len(x), "model rmse", round(np.sqrt(((x.pred-x.result)**2).mean()),2), "market", round(np.sqrt(((x.spread_line-x.result)**2).mean()),2), "ats", round(ats(x.pred.values,x.spread_line.values,x.result.values)[0],3))
# totals
t = d[d.total_line.notna()]
pt = t.proj_home_pts + t.proj_away_pts
print("total rmse model", np.sqrt(((pt - t.total)**2).mean()), "market", np.sqrt(((t.total_line - t.total)**2).mean()), "bias", (pt-t.total).mean())
