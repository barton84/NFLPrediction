"""Point-in-time team ratings.

For every game, each team's rating is computed using ONLY games played before
that week. Ratings are opponent-adjusted (a ridge regression of each game's
home-minus-away stat on team strengths) and shrunk toward a prior built from
last season's final rating times a carryover factor. Early in the year the
prior dominates; as games accumulate the current season takes over.

Signals (all expressed as home-minus-away, per game):
  margin   points margin
  epa      EPA/play net (offense EPA/play minus EPA/play allowed), close-game plays only
  sr       success rate net
  ypp      yards/play net
  pass_epa dropback EPA/play net
"""
import numpy as np, pandas as pd

TEAMS = None
SIGNALS = ["margin", "epa", "sr", "ypp", "pass_epa"]


def game_signals(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=g.index)
    out["margin"] = g.result
    out["epa"] = g.home_epa - g.away_epa
    out["sr"] = g.home_sr - g.away_sr
    out["ypp"] = g.home_ypp - g.away_ypp
    out["pass_epa"] = g.home_pass_epa - g.away_pass_epa
    return out


def ridge_net(games, y, teams, prior, lam, hfa):
    """Solve y = r_home - r_away + hfa*nonneutral, ridge-shrunk toward prior."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    A = np.zeros((len(games) + n, n))
    b = np.zeros(len(games) + n)
    for row, (h, a, neu, yy) in enumerate(zip(games.home_team, games.away_team, games.neutral, y)):
        A[row, idx[h]] = 1
        A[row, idx[a]] = -1
        b[row] = yy - (0 if neu else hfa)
    s = np.sqrt(lam)
    A[len(games):, :] = np.eye(n) * s
    b[len(games):] = np.array([prior.get(t, 0.0) for t in teams]) * s
    # pin the mean of ratings to zero via one extra soft row
    A = np.vstack([A, np.ones(n) * 10])
    b = np.append(b, 0)
    r = np.linalg.lstsq(A, b, rcond=None)[0]
    return dict(zip(teams, r))


def ridge_offdef(games, teams, prior_o, prior_d, lam, mu, hfa):
    """Points scored model: pts = mu + off_T - def_O (+hfa/2 at home, -hfa/2 away).
    def is 'points prevented', higher = better defense."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rows, b = [], []
    for h, a, neu, hs, as_ in zip(games.home_team, games.away_team, games.neutral, games.home_score, games.away_score):
        for T, O, pts, sgn in [(h, a, hs, 1), (a, h, as_, -1)]:
            v = np.zeros(2 * n)
            v[idx[T]] = 1
            v[n + idx[O]] = -1
            rows.append(v)
            b.append(pts - mu - (0 if neu else sgn * hfa / 2))
    A = np.array(rows) if rows else np.zeros((0, 2 * n))
    b = np.array(b)
    s = np.sqrt(lam)
    P = np.eye(2 * n) * s
    pb = np.concatenate([[prior_o.get(t, 0) for t in teams], [prior_d.get(t, 0) for t in teams]]) * s
    cons = np.zeros((2, 2 * n)); cons[0, :n] = 10; cons[1, n:] = 10
    A = np.vstack([A, P, cons]); b = np.concatenate([b, pb, [0, 0]])
    r = np.linalg.lstsq(A, b, rcond=None)[0]
    return dict(zip(teams, r[:n])), dict(zip(teams, r[n:]))


def build(g: pd.DataFrame, carry=0.5, k=6.0, final_lam=1.0, wt_prior=None, wt_blend=1.0, k_margin=None):
    """Returns (features per game, ratings snapshots keyed by (season, week))."""
    g = g.copy()
    g["neutral"] = (g.location == "Neutral").astype(int)
    sig = game_signals(g)
    g = g.join(sig)
    teams = sorted(set(g.home_team) | set(g.away_team))
    seasons = sorted(g.season.unique())
    done = g[g.result.notna() & g.home_epa.notna()]

    # home-field per signal and league scoring avg, from the previous two seasons
    def league_consts(season):
        d = done[(done.season < season) & (done.season >= season - 2) & (done.neutral == 0)]
        if len(d) == 0:
            d = done[(done.season == season) & (done.neutral == 0)]
        hfa = {s: d[s].mean() for s in SIGNALS}
        mu = (d.home_score.mean() + d.away_score.mean()) / 2
        return hfa, mu

    prior = {s: {} for s in SIGNALS}
    prior_o, prior_d = {}, {}
    feats, snaps = [], {}
    for season in seasons:
        if wt_prior is not None:   # starting margin rating from the market's season win total
            base = prior["margin"]
            prior["margin"] = {t: wt_blend * wt_prior.get((season, t), base.get(t, 0.0)) + (1 - wt_blend) * base.get(t, 0.0)
                               for t in teams}
        gs = g[g.season == season]
        hfa, mu = league_consts(season)
        for week in sorted(gs.week.unique()):
            past = done[(done.season == season) & (done.week < week)]
            rat = {s: ridge_net(past, past[s].values, teams, prior[s], (k_margin or k) if s == "margin" else k, hfa[s]) for s in SIGNALS}
            ro, rd = ridge_offdef(past, teams, prior_o, prior_d, k, mu, hfa["margin"])
            snaps[(season, week)] = {"net": rat, "off": ro, "def": rd, "hfa": hfa, "mu": mu,
                                     "games_played": past.groupby("home_team").size().add(
                                         past.groupby("away_team").size(), fill_value=0).to_dict()}
            wk = gs[gs.week == week]
            for i, r in wk.iterrows():
                f = {"game_id": r.game_id, "season": season, "week": week}
                for s in SIGNALS:
                    f[f"d_{s}"] = rat[s][r.home_team] - rat[s][r.away_team]
                f["d_off_def"] = (ro[r.home_team] - rd[r.away_team]) - (ro[r.away_team] - rd[r.home_team])
                f["proj_home_pts"] = mu + ro[r.home_team] - rd[r.away_team]
                f["proj_away_pts"] = mu + ro[r.away_team] - rd[r.home_team]
                f["hfa_margin"] = hfa["margin"]
                feats.append(f)
        # end of season: final ratings become next season's prior (times carry)
        full = done[done.season == season]
        if len(full):
            for s in SIGNALS:
                fin = ridge_net(full, full[s].values, teams, {}, final_lam, hfa[s])
                prior[s] = {t: v * carry for t, v in fin.items()}
            fo, fd = ridge_offdef(full, teams, {}, {}, final_lam, mu, hfa["margin"])
            prior_o = {t: v * carry for t, v in fo.items()}
            prior_d = {t: v * carry for t, v in fd.items()}
    F = pd.DataFrame(feats)
    out = g.merge(F, on=["game_id", "season", "week"])
    return out, snaps
