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

import fetch_data, build_games, fetch_odds, injuries, weather, line_log
import probability as P
from ratings import build
from qb import add_qb_features, load_dropbacks, qb_rating_before, REPLACEMENT, N0, DECAY
from backtest import prep, loso, report, ats

CARRY, K = 0.5, 8          # prior carryover from last season, prior weight in games
MODEL_VERSION = "v2.0"     # bump whenever the model math changes, so line movement can be compared by version
FEATURES = ["home", "d_margin", "d_sr", "d_qb_delta", "rest_diff"]
T_FEATURES = ["proj", "dome", "wind10", "cold32", "qbsum"]
PRECIP_PRIOR = {"rain": -2.0, "snow": -3.0}   # untested, applied only when the forecast says rain or snow is likely


def total_features(x, wx=None):
    """Totals model inputs. wx = forecast dict (this week) or None to use recorded game weather."""
    x = x.copy()
    x["proj"] = x.proj_home_pts + x.proj_away_pts
    indoor = x.roof.isin(["dome", "closed"])
    x["dome"] = indoor.astype(int)
    x["wind10"] = np.where(indoor, 0, (x.wind.fillna(0).clip(upper=22) - 10).clip(lower=0))
    x["cold32"] = np.where(indoor, 0, (32 - x.temp.fillna(60)).clip(lower=0))
    x["qbsum"] = x.home_qb_delta.fillna(0) + x.away_qb_delta.fillna(0)
    return x
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
    if os.environ.get("FORCE_WEEK"):   # testing only: rebuild as if an earlier week were upcoming
        this_week = int(os.environ["FORCE_WEEK"]); last_week = this_week - 1

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

    # ---------- cover probability: key-number distribution, lean factor fitted per holdout ----------
    d["edge"] = d.pred - d.spread_line
    kw = P.key_weights(d.result)
    grid = np.round(np.arange(0, 0.41, 0.05), 2)
    k_folds, calib_rows = {}, []
    for s_ in test_seasons:
        tr, te = d[d.season != s_], d[d.season == s_]
        kwf = P.key_weights(tr.result)
        Kf = P.fit_k(tr.edge.values, tr.spread_line.values, tr.result.values, kwf, grid=grid)
        k_folds[int(s_)] = Kf
        for r in te.itertuples():
            if r.result == r.spread_line:
                continue
            a, pp = P.home_cover(r.spread_line, P.market_center(r.spread_line, kwf) + Kf * r.edge, kwf)
            home_side = r.edge > 0
            calib_rows.append((a if home_side else 1 - a - pp, (r.result > r.spread_line) == home_side))
    K_SPREAD = P.fit_k(d.edge.values, d.spread_line.values, d.result.values, kw, grid=grid)
    cr = pd.DataFrame(calib_rows, columns=["p", "won"])
    cr["b"] = pd.cut(cr.p, [0, 0.49, 0.51, 0.53, 1], labels=["Under 49%", "49 to 51%", "51 to 53%", "53%+"])
    calib = [{"bucket": str(b), "n": int(len(x)), "said": nz(x.p.mean(), 3), "hit": nz(x.won.mean(), 3)}
             for b, x in cr.groupby("b", observed=True)]
    key_freq = {int(k): nz((d.result.abs() == k).mean(), 4) for k in range(1, 18)}

    # ---------- totals model ----------
    dt = total_features(d[d.total_line.notna()])
    tpred = pd.Series(index=dt.index, dtype=float); tco = []
    for s_ in test_seasons:
        tr, te = dt[dt.season != s_], dt[dt.season == s_]
        mt = LinearRegression().fit(tr[T_FEATURES], tr.total)
        tpred[te.index] = mt.predict(te[T_FEATURES]); tco.append(dict(zip(T_FEATURES, mt.coef_)))
    tco = pd.DataFrame(tco)
    te_ = tpred - dt.total_line; tr_ = dt.total - dt.total_line; dec = tr_ != 0
    t_hit = float((np.sign(te_) == np.sign(tr_))[dec].mean()); big = dec & (te_.abs() >= 5)
    MT = LinearRegression().fit(dt[T_FEATURES], dt.total)
    TC = dict(zip(T_FEATURES, [float(v) for v in MT.coef_])); TC["intercept"] = float(MT.intercept_)
    K_TOTAL = P.fit_k(te_.values, dt.total_line.values, dt.total.values, None, grid=grid, total=True)
    totals_info = {"coef": TC, "ranges": {f: [nz(tco[f].min(), 3), nz(tco[f].max(), 3)] for f in T_FEATURES},
                   "rmse": nz(np.sqrt(((tpred - dt.total) ** 2).mean()), 2),
                   "market_rmse": nz(np.sqrt(((dt.total_line - dt.total) ** 2).mean()), 2),
                   "ou_hit": nz(t_hit, 3), "hit5": nz(float((np.sign(te_) == np.sign(tr_))[big].mean()), 3), "n5": int(big.sum()),
                   "K": K_TOTAL, "sd": P.TOTAL_SD, "precip_prior": PRECIP_PRIOR}
    print("cover K folds", k_folds, "pooled", K_SPREAD, "| totals rmse", totals_info["rmse"], "K", K_TOTAL)

    # ---------- frozen weights: the daily run reads them from a file so they only change on purpose ----------
    # Retrain once a year (or after a deliberate model change) with:  RETRAIN=1 python pipeline/run.py
    # then bump MODEL_VERSION and commit pipeline/model_weights.json.
    wpath = os.path.join(HERE, "model_weights.json")
    if os.path.exists(wpath) and not os.environ.get("RETRAIN"):
        saved = json.load(open(wpath))
        W.update(saved["spread"]); TC.update(saved["totals"])
        K_SPREAD, K_TOTAL, kw = saved["K_spread"], saved["K_total"], np.array(saved["key_weights"])
        weights.update({"hfa": W["home"], "margin": W["d_margin"], "sr": W["d_sr"], "qb": W["d_qb_delta"], "rest": W["rest_diff"],
                        "ranges": saved["spread_ranges"]})
        totals_info.update({"coef": TC, "K": K_TOTAL, "ranges": saved["totals_ranges"]})
        print("using frozen weights from", saved["trained"], "version", saved["version"])
    else:
        json.dump({"version": MODEL_VERSION, "trained": datetime.date.today().isoformat(),
                   "trained_on": [min(test_seasons), max(test_seasons)], "games": int(len(d)),
                   "spread": W, "spread_ranges": weights["ranges"], "totals": TC, "totals_ranges": totals_info["ranges"],
                   "K_spread": K_SPREAD, "K_total": K_TOTAL, "key_weights": [round(float(v), 6) for v in kw]},
                  open(wpath, "w"), indent=1)
        print("trained and saved weights to", wpath)
    weights["trained"] = json.load(open(wpath))["trained"]

    def total_model(x, wx_list=None):
        f = total_features(x)
        if wx_list is not None:  # forecast overrides recorded weather for upcoming games
            for i, wx in zip(f.index, wx_list):
                if wx and not wx.get("indoor"):
                    f.loc[i, "wind10"] = max(0, min(wx["wind"], 22) - 10)
                    f.loc[i, "cold32"] = max(0, 32 - wx["temp"])
                    f.loc[i, "dome"] = 0
                elif wx and wx.get("indoor"):
                    f.loc[i, ["wind10", "cold32"]] = 0; f.loc[i, "dome"] = 1
        base = f[T_FEATURES].values @ np.array([TC[k] for k in T_FEATURES]) + TC["intercept"]
        return base, f

    def precip_adj(wx):
        if not wx or wx.get("indoor"):
            return 0.0, None
        if wx.get("snow_in", 0) >= 0.5:
            return PRECIP_PRIOR["snow"], "snow"
        if wx.get("precip_prob", 0) >= 60 and wx.get("precip_in", 0) >= 0.1:
            return PRECIP_PRIOR["rain"], "rain"
        return 0.0, None

    def cover_probs(line, model, total, tmodel):
        out = {}
        if line is not None and not pd.isna(line):
            c = P.market_center(line, kw) + K_SPREAD * (model - line)
            a, pp = P.home_cover(line, c, kw)
            home_side = model > line
            out.update(cover=nz(a if home_side else 1 - a - pp, 3), push=nz(pp, 3), side_home=bool(home_side))
        if total is not None and not pd.isna(total):
            o, pp = P.over_prob(total, total + K_TOTAL * (tmodel - total))
            over_side = tmodel > total
            out.update(t_prob=nz(o if over_side else 1 - o - pp, 3), t_push=nz(pp, 3), t_side="Over" if over_side else "Under")
        return out

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

    # ---------- last week + season to date ----------
    def graded(week):
        x = df[(df.season == season) & (df.week == week)].copy()
        if x.empty:
            return []
        x["model"] = predict(x)
        x["t_model"], _ = total_model(x)
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
            row.update(total=nz(total, 1), t_model=nz(r.t_model, 1))
            if pd.notna(r.result) and total is not None and not pd.isna(total):
                act = r.home_score + r.away_score
                row["t_actual"] = nz(act, 0)
                side, res = np.sign(r.t_model - total), np.sign(act - total)
                row["t_ats"] = "push" if res == 0 else ("win" if side == res else "loss")
                row["t_side"] = "Over" if r.t_model > total else "Under"
                row["t_edge"] = nz(r.t_model - total, 1)
            out.append(row)
        return out

    season_weeks = []
    for w in range(1, last_week + 1):
        rows = graded(w)
        dec = [r for r in rows if r.get("ats") in ("win", "loss")]
        big = [r for r in dec if abs(r["edge"]) >= 5]
        errs = [abs(r["error"]) for r in rows if "error" in r]
        mkt = [abs(r["result"] - r["line"]) for r in rows if "error" in r]
        tdec = [r for r in rows if r.get("t_ats") in ("win", "loss")]
        season_weeks.append({"week": w, "wins": sum(r["ats"] == "win" for r in dec), "losses": sum(r["ats"] == "loss" for r in dec),
                             "t_wins": sum(r["t_ats"] == "win" for r in tdec), "t_losses": sum(r["t_ats"] == "loss" for r in tdec),
                             "big_wins": sum(r["ats"] == "win" for r in big), "big_losses": sum(r["ats"] == "loss" for r in big),
                             "mae": nz(np.mean(errs), 1) if errs else None, "market_mae": nz(np.mean(mkt), 1) if mkt else None})

    # ---------- team ratings for the matchup tool and table ----------
    snap = snaps[(season, this_week)]
    played = df[(df.season == season) & df.result.notna() & (df.week < this_week)]
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
        allg = df[((df.home_team == t) | (df.away_team == t)) & df.result.notna() & ((df.season < season) | (df.week < this_week))].sort_values(["season", "week"]).tail(8)
        starters = [r.home_qb_id if r.home_team == t else r.away_qb_id for r in allg.itertuples()]
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

    # ---------- this week: injuries, weather, model, probabilities ----------
    if fetch:
        injuries.fetch(season)
    inj, dc = injuries.load(season, this_week)
    rate = lambda gid: (qb_rating_before(db, gid, season, this_week)[0] if isinstance(gid, str) else REPLACEMENT)
    tw = df[(df.season == season) & (df.week == this_week)].copy().sort_values(["gameday", "gametime"])
    reports, wx_list = {}, []
    for r in tw.itertuples():
        rep = {}
        for side, team, qid, qname in [("home", r.home_team, r.home_qb_id, r.home_qb_name), ("away", r.away_team, r.away_qb_id, r.away_qb_name)]:
            rep[side] = injuries.team_report(team, qid, qname, inj, dc, rate)
            q = rep[side]["qb"]; base = teams[team]["qb_base"]
            r_start = rate(qid)
            eff = r_start
            if q.get("status"):
                rb = q["rating_backup"] if q.get("rating_backup") is not None else REPLACEMENT
                eff = q["play_prob"] * r_start + (1 - q["play_prob"]) * rb
                q["rating_start"], q["rating_backup"] = nz(r_start, 3), nz(rb, 3)
            # usual starter hurt but the schedule already lists his replacement: note it, nothing to adjust
            prev_id, prev_name = teams[team]["qb_id"], teams[team]["qb"]
            if not q.get("status") and isinstance(prev_id, str) and prev_id != qid and len(inj):
                st = inj[(inj.team == team) & (inj.gsis_id == prev_id)].report_status
                if len(st):
                    q["replaced"] = {"name": prev_name, "status": st.iloc[0]}
            rep[side]["qb_delta_sched"] = nz(r_start - base, 3)
            rep[side]["qb_delta"] = nz(eff - base, 3)
        reports[r.game_id] = rep
        wx_list.append(weather.forecast({"id": r.game_id, "stadium_id": r.stadium_id, "stadium": r.stadium,
                                         "roof": r.roof if isinstance(r.roof, str) else None, "date": r.gameday, "time": r.gametime}))
    tw["home_qb_delta"] = [reports[i]["home"]["qb_delta"] for i in tw.game_id]
    tw["away_qb_delta"] = [reports[i]["away"]["qb_delta"] for i in tw.game_id]
    tw["d_qb_delta"] = tw.home_qb_delta - tw.away_qb_delta
    tw["model"] = predict(tw)
    tbase, tf = total_model(tw, wx_list)
    games = []
    for (r, tb, wx, fr) in zip(tw.itertuples(), tbase, wx_list, tf.itertuples()):
        line, total, src = line_info(r, this_week)
        padj, ptype = precip_adj(wx)
        tmodel = tb + padj
        g_ = {
            "id": r.game_id, "away": r.away_team, "home": r.home_team, "date": r.gameday, "time": r.gametime,
            "season": int(r.season), "week": int(r.week), "weekday": r.weekday, "roof": r.roof if isinstance(r.roof, str) else None, "stadium": r.stadium,
            "neutral": int(r.neutral), "rest_home": nz(r.home_rest, 0), "rest_away": nz(r.away_rest, 0),
            "home_qb": r.home_qb_name, "away_qb": r.away_qb_name,
            "home_qb_delta": reports[r.game_id]["home"]["qb_delta"] or 0, "away_qb_delta": reports[r.game_id]["away"]["qb_delta"] or 0,
            "line": nz(line, 1), "total": nz(total, 1), "line_src": src,
            "model": nz(r.model, 2), "t_model": nz(tmodel, 1), "precip": ptype, "dome": int(fr.dome),
            "edge": nz(r.model - line, 2) if line is not None and not pd.isna(line) else None,
            "t_edge": nz(tmodel - total, 1) if total is not None and not pd.isna(total) else None,
            "result": nz(r.result, 0), "home_score": nz(r.home_score, 0), "away_score": nz(r.away_score, 0),
            "inj": reports[r.game_id], "wx": wx,
        }
        g_.update(cover_probs(line, r.model, total, tmodel))
        games.append(g_)
    n_wx = sum(1 for w in wx_list if w and not w.get("indoor"))

    # ---------- line movement log (permanent, one snapshot per game per day) ----------
    log_path = os.path.join(OUT, "line_log.json")
    log = line_log.load(log_path)
    if not os.environ.get("FORCE_WEEK"):
        log = line_log.record(log, games, MODEL_VERSION)
    fin = df[df.game_id.isin(log.keys()) & df.result.notna()]
    log = line_log.settle(log, {r.game_id: (float(r.spread_line), float(r.result)) for r in fin.itertuples() if pd.notna(r.spread_line)})
    if not os.environ.get("FORCE_WEEK"):
        with open(log_path, "w") as f_:
            json.dump(log, f_, indent=1, sort_keys=True)
    clv_rows = line_log.clv_rows(log)
    clv = {"rows": [r for r in clv_rows if r["season"] == season], "summary": line_log.summarize([r for r in clv_rows if r["season"] == season]),
           "tracked_games": len(log), "version": MODEL_VERSION}
    print(f"line log: {len(log)} games tracked, {len(clv_rows)} graded")
    print(f"injury report rows {len(inj)}, depth chart teams {dc.team.nunique() if len(dc) else 0}, forecasts {n_wx}")

    out = {
        "meta": {"season": season, "this_week": this_week, "last_week": last_week, "built": now,
                 "games_in_backtest": int(len(d)), "backtest_seasons": [min(test_seasons), max(test_seasons)],
                 "carry": CARRY, "prior_games": K, "model_version": MODEL_VERSION, "replacement_qb": REPLACEMENT,
                 "mu": nz(snap["mu"], 3), "dk_lines_this_week": sum(1 for x in games if x["line_src"] == "DK"), "hfa_note": snap["hfa"]["margin"]},
        "weights": weights, "teams": teams, "qbs": qbs, "this_week": games,
        "last_week": graded(last_week) if last_week else [], "season_weeks": season_weeks,
        "backtest": {"variants": bt_rows, "by_season": by_season, "market_rmse": nz(market_rmse, 2)},
        "prob": {"kw": [round(float(v), 4) for v in kw], "k_min": int(P.KS[0]), "sd": P.SPREAD_SD, "K": K_SPREAD,
                 "K_folds": k_folds, "calib": calib, "key_freq": key_freq},
        "totals": totals_info,
        "clv": clv,
        "feeds": {"injury_report": bool(len(inj)), "depth_chart": bool(len(dc)), "forecasts": n_wx},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "model.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"), default=lambda o: None)
    print(f"wrote model.json: season {season} this_week {this_week} last_week {last_week}; weights", {k: round(v, 3) for k, v in W.items()})


if __name__ == "__main__":
    main(fetch="--no-fetch" not in sys.argv)
