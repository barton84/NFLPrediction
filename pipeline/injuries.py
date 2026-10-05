"""Injury feed: official NFL injury reports + latest depth charts (both via nflverse).

QB status feeds the model automatically, because the QB adjustment is the
one injury effect that has been backtested. If the projected starter is
listed Out, his backup (QB2 on the depth chart) is rated instead. Doubtful
and Questionable blend the two QBs by how often players with that tag
usually play.

Other injured starters are listed with rough position values. Those values
are untested priors, so they only count if you switch them on in the app.
"""
import os, urllib.request
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")
INJ = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{y}.csv"
DC = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{y}.parquet"

PLAY_PROB = {"Out": 0.0, "Doubtful": 0.2, "Questionable": 0.8}
# Untested position priors, in points of spread, for a starter who misses the game
POS_VALUE = {
    "WR": 0.6, "TE": 0.4, "RB": 0.3,
    "LT": 0.6, "RT": 0.3, "LG": 0.3, "RG": 0.3, "C": 0.3,
    "LDE": 0.5, "RDE": 0.5, "LOLB": 0.5, "ROLB": 0.5,
    "LDT": 0.3, "RDT": 0.3, "NT": 0.3,
    "WLB": 0.2, "MLB": 0.2, "SLB": 0.2, "LILB": 0.2, "RILB": 0.2,
    "LCB": 0.5, "RCB": 0.5, "NB": 0.2, "FS": 0.3, "SS": 0.3,
}
POS_LABEL = {"LT": "Left tackle", "RT": "Right tackle", "LG": "Left guard", "RG": "Right guard", "C": "Center",
             "LDE": "Edge", "RDE": "Edge", "LOLB": "Edge", "ROLB": "Edge", "LDT": "DT", "RDT": "DT", "NT": "Nose tackle",
             "WLB": "LB", "MLB": "LB", "SLB": "LB", "LILB": "LB", "RILB": "LB", "LCB": "CB", "RCB": "CB",
             "NB": "Nickel CB", "FS": "Safety", "SS": "Safety"}


def fetch(season):
    for url, name in [(INJ.format(y=season), f"injuries_{season}.csv"), (DC.format(y=season), f"depth_{season}.parquet")]:
        try:
            print("fetch", url)
            urllib.request.urlretrieve(url, os.path.join(RAW, name))
        except Exception as e:
            print("injury/depth fetch failed:", e)


def load(season, week):
    """Returns (report rows for this week, latest depth chart) or empty frames."""
    ip, dp = os.path.join(RAW, f"injuries_{season}.csv"), os.path.join(RAW, f"depth_{season}.parquet")
    inj = pd.read_csv(ip) if os.path.exists(ip) else pd.DataFrame()
    if len(inj):
        inj = inj[(inj.week == week) & inj.report_status.isin(PLAY_PROB.keys())]
    dc = pd.read_parquet(dp) if os.path.exists(dp) else pd.DataFrame()
    if len(dc):
        dc = dc[dc.dt == dc.groupby("team").dt.transform("max")]
    return inj, dc


def team_report(team, sched_qb_id, sched_qb_name, inj, dc, rate):
    """rate(gsis_id) -> QB EPA/dropback rating. Returns a dict for the app."""
    out = {"qb": {"name": sched_qb_name, "id": sched_qb_id, "status": None}, "players": [], "report": bool(len(inj))}
    tinj = inj[inj.team == team] if len(inj) else inj
    tdc = dc[dc.team == team] if len(dc) else dc
    status = dict(zip(tinj.gsis_id, tinj.report_status)) if len(tinj) else {}

    # ----- quarterback -----
    qbs = tdc[tdc.pos_abb == "QB"].sort_values("pos_rank") if len(tdc) else pd.DataFrame()
    st = status.get(sched_qb_id)
    if st:
        backup = None
        for r in qbs.itertuples():
            if r.gsis_id != sched_qb_id and status.get(r.gsis_id) != "Out":
                backup = r; break
        p = PLAY_PROB[st]
        r_start = rate(sched_qb_id)
        r_back = rate(backup.gsis_id) if backup is not None else None
        out["qb"] = {"name": sched_qb_name, "id": sched_qb_id, "status": st, "play_prob": p,
                     "backup": backup.player_name if backup is not None else "Unknown backup",
                     "backup_id": backup.gsis_id if backup is not None else None,
                     "rating_start": r_start, "rating_backup": r_back}
    # ----- other starters on the report -----
    if len(tdc) and len(tinj):
        starters = tdc[(tdc.pos_rank == 1) & tdc.pos_abb.isin(POS_VALUE.keys())]
        seen = set()
        for r in starters.itertuples():
            stt = status.get(r.gsis_id)
            if not stt or r.gsis_id in seen:
                continue
            seen.add(r.gsis_id)
            v = POS_VALUE[r.pos_abb] * (1 - PLAY_PROB[stt])
            out["players"].append({"name": r.player_name, "pos": POS_LABEL.get(r.pos_abb, r.pos_abb),
                                   "status": stt, "pts": round(v, 2)})
        out["players"].sort(key=lambda x: -x["pts"])
    return out
