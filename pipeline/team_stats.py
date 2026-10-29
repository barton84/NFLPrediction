"""Season-to-date team cards for Game Detail: record, against the spread,
over/under, and offense / defense per game with league ranks.

ATS and over/under use the nflverse consensus closing line, so every game is
graded the same way. Pass yards are net of sacks, like official team stats.
Ranks run 1 to 32, where 1 is best (most gained on offense, fewest allowed on
defense).
"""
import os
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")
REMAP = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def build(season):
    g = pd.read_csv(os.path.join(RAW, "games.csv"))
    g = g[(g.season == season) & (g.game_type == "REG") & g.result.notna()].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(REMAP)
    if g.empty:
        return {}
    # one row per team per game
    rows = []
    for r in g.itertuples():
        for team, opp, pf, pa, home in ((r.home_team, r.away_team, r.home_score, r.away_score, True),
                                        (r.away_team, r.home_team, r.away_score, r.home_score, False)):
            margin = pf - pa
            ats = None
            if pd.notna(r.spread_line):   # spread_line: points the home team is favored by
                cover_m = (r.result - r.spread_line) * (1 if home else -1)
                ats = "w" if cover_m > 0 else "l" if cover_m < 0 else "p"
            ou = None
            if pd.notna(r.total_line):
                tot = r.home_score + r.away_score
                ou = "o" if tot > r.total_line else "u" if tot < r.total_line else "p"
            rows.append({"game_id": r.game_id, "team": team, "pf": pf, "pa": pa,
                         "wl": "w" if margin > 0 else "l" if margin < 0 else "t", "ats": ats, "ou": ou})
    t = pd.DataFrame(rows)

    # yards from play-by-play
    path = os.path.join(RAW, f"pbp_{season}.parquet")
    yd = pd.DataFrame(columns=["game_id", "team", "pass_yds", "rush_yds"])
    if os.path.exists(path):
        p = pd.read_parquet(path, columns=["game_id", "posteam", "passing_yards", "rushing_yards", "sack", "yards_gained"])
        p = p[p.posteam.notna() & p.game_id.isin(g.game_id)]
        p["posteam"] = p.posteam.replace(REMAP)
        sack_yds = np.where(p.sack.fillna(0) == 1, p.yards_gained.fillna(0), 0)
        p = p.assign(pass_net=p.passing_yards.fillna(0) + sack_yds, rush=p.rushing_yards.fillna(0))
        yd = p.groupby(["game_id", "posteam"]).agg(pass_yds=("pass_net", "sum"), rush_yds=("rush", "sum")).reset_index() \
              .rename(columns={"posteam": "team"})
    t = t.merge(yd, on=["game_id", "team"], how="left")
    # what each team allowed = what its opponent gained in the same game
    opp = t[["game_id", "team", "pass_yds", "rush_yds"]].rename(columns={"team": "opp", "pass_yds": "pass_alw", "rush_yds": "rush_alw"})
    t = t.merge(t[["game_id", "team"]].rename(columns={"team": "opp"}), on="game_id")
    t = t[t.team != t.opp].merge(opp, on=["game_id", "opp"], how="left")

    agg = t.groupby("team").agg(
        gp=("game_id", "size"), w=("wl", lambda s: (s == "w").sum()), l=("wl", lambda s: (s == "l").sum()),
        tie=("wl", lambda s: (s == "t").sum()),
        ats_w=("ats", lambda s: (s == "w").sum()), ats_l=("ats", lambda s: (s == "l").sum()), ats_p=("ats", lambda s: (s == "p").sum()),
        ou_o=("ou", lambda s: (s == "o").sum()), ou_u=("ou", lambda s: (s == "u").sum()), ou_p=("ou", lambda s: (s == "p").sum()),
        pts=("pf", "mean"), pass_yds=("pass_yds", "mean"), rush_yds=("rush_yds", "mean"),
        pts_alw=("pa", "mean"), pass_alw=("pass_alw", "mean"), rush_alw=("rush_alw", "mean"))
    for c in ("pts", "pass_yds", "rush_yds"):
        agg[c + "_rk"] = agg[c].rank(ascending=False, method="min")
    for c in ("pts_alw", "pass_alw", "rush_alw"):
        agg[c + "_rk"] = agg[c].rank(ascending=True, method="min")
    out = {}
    for team, r in agg.iterrows():
        d = {k: (int(v) if k.endswith("_rk") or k in ("gp", "w", "l", "tie", "ats_w", "ats_l", "ats_p", "ou_o", "ou_u", "ou_p")
                 else round(float(v), 1)) for k, v in r.items() if pd.notna(v)}
        out[team] = d
    return out


if __name__ == "__main__":
    import json, sys
    s = build(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
    print(len(s), "teams"); print(json.dumps({k: s[k] for k in list(s)[:2]}, indent=1))
