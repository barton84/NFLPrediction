"""Point-in-time QB ratings and a per-game 'QB change' feature.

QB rating = shrunk EPA per dropback using every prior dropback since 2015
(older seasons decayed), pulled toward a replacement-level value so a backup
with 40 attempts is not treated like an established starter.

qb_delta for a team in a game = rating(today's starter) minus the average
rating of the starters in that team's previous 8 games. Positive = upgrade at
QB relative to what the team's recent results were built on.
"""
import pandas as pd, numpy as np, glob, os

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")
REPLACEMENT = -0.12   # EPA/dropback of a typical backup
N0 = 250              # dropbacks of prior weight
DECAY = 0.6           # weight per season back


def load_dropbacks():
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW, "pbp_*.parquet"))):
        d = pd.read_parquet(f, columns=["game_id", "season", "week", "passer_player_id", "qb_epa", "qb_dropback", "game_date"])
        d = d[(d.qb_dropback == 1) & d.passer_player_id.notna() & d.qb_epa.notna()]
        fr.append(d.groupby(["season", "week", "game_id", "passer_player_id"]).agg(
            epa=("qb_epa", "sum"), n=("qb_epa", "size"), date=("game_date", "first")).reset_index())
    out = pd.concat(fr)
    out["game_id"] = out.game_id.str.replace("_STL", "_LA").str.replace("_SD", "_LAC").str.replace("_OAK", "_LV")
    return out


def qb_rating_before(db, qb, season, week):
    h = db[(db.passer_player_id == qb) & ((db.season < season) | ((db.season == season) & (db.week < week)))]
    if h.empty:
        return REPLACEMENT, 0
    w = DECAY ** (season - h.season)
    epa, n = (h.epa * w).sum(), (h.n * w).sum()
    return (epa + REPLACEMENT * N0) / (n + N0), int(h.n.sum())


def add_qb_features(g: pd.DataFrame) -> pd.DataFrame:
    db = load_dropbacks()
    db = db.sort_values(["season", "week"])
    # precompute ratings for every (qb, season, week) needed
    cache = {}

    def rate(qb, s, w):
        if pd.isna(qb):
            return np.nan
        key = (qb, s, w)
        if key not in cache:
            cache[key] = qb_rating_before(db, qb, s, w)[0]
        return cache[key]

    g = g.sort_values(["season", "week"]).copy()
    hist = {}  # team -> list of (season, week, starter rating at that time)
    hq, aq, hd, ad = [], [], [], []
    for r in g.itertuples():
        row = {}
        for side, qb, team in [("h", r.home_qb_id, r.home_team), ("a", r.away_qb_id, r.away_team)]:
            cur = rate(qb, r.season, r.week)
            past = [x for x in hist.get(team, [])][-8:]
            base = np.mean([rate(q, r.season, r.week) for q in past]) if past else cur
            row[side] = (cur, (cur - base) if not np.isnan(cur) else 0.0)
        hq.append(row["h"][0]); aq.append(row["a"][0]); hd.append(row["h"][1]); ad.append(row["a"][1])
        if pd.notna(r.result):
            hist.setdefault(r.home_team, []).append(r.home_qb_id)
            hist.setdefault(r.away_team, []).append(r.away_qb_id)
    g["home_qb_rating"], g["away_qb_rating"] = hq, aq
    g["home_qb_delta"], g["away_qb_delta"] = hd, ad
    g["d_qb_delta"] = g.home_qb_delta - g.away_qb_delta
    return g
