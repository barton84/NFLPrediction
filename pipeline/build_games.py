"""Step 1: turn nflverse schedules + play-by-play into one row per game with
home/away efficiency stats. Output: raw/game_stats.parquet
"""
import pandas as pd, numpy as np, glob, os

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")
INTL_STADIUMS = {"Wembley Stadium", "Twickenham Stadium", "Tottenham Stadium", "Tottenham Hotspur Stadium",
                 "Azteca Stadium", "Estadio Banorte", "Allianz Arena", "FC Bayern Munich Stadium", "Deutsche Bank Park",
                 "Arena Corinthians", "Bernabeu", "Melbourne Cricket Ground", "Stade de France", "Maracana Stadium"}
INTL_IDS = {"LON00", "LON01", "LON02", "MEX00", "GER00", "MUN01", "FRA00", "SAO00", "MAD01", "MEL00", "PAR00", "RIO00"}


def team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna()]
    p = p[(p.qb_kneel != 1) & (p.qb_spike != 1) & p.posteam.notna()]
    # garbage-time filter: drop plays where the game is basically decided
    p = p.assign(close=((p.wp >= 0.05) & (p.wp <= 0.95)).astype(int))
    pc = p[p.close == 1]
    agg = lambda d, sfx: d.groupby(["game_id", "posteam"]).agg(
        **{f"epa{sfx}": ("epa", "mean"), f"sr{sfx}": ("success", "mean"), f"n{sfx}": ("epa", "size")})
    a = agg(pc, "")
    b = p.groupby(["game_id", "posteam"]).agg(ypp=("yards_gained", "mean"), plays=("epa", "size"),
                                              pass_epa=("epa", lambda s: s[p.loc[s.index, "pass"] == 1].mean()))
    return a.join(b).reset_index().rename(columns={"posteam": "team"})


def main():
    g = pd.read_csv(os.path.join(RAW, "games.csv"))
    g = g[g.season >= 2015].copy()
    # relocated franchises: use one code per franchise across all seasons
    remap = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
    for c in ["home_team", "away_team"]:
        g[c] = g[c].replace(remap)
    g["game_id"] = g.game_id.str.replace("_STL", "_LA").str.replace("_SD", "_LAC").str.replace("_OAK", "_LV")
    # International games are neutral sites even when the schedule lists a "home" team
    intl = g.stadium.isin(INTL_STADIUMS) | g.stadium_id.isin(INTL_IDS)
    g.loc[intl, "location"] = "Neutral"
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "pbp_*.parquet"))):
        cols = ["game_id", "posteam", "play_type", "epa", "success", "yards_gained", "qb_kneel", "qb_spike", "wp", "pass"]
        d = pd.read_parquet(f, columns=cols)
        d["game_id"] = d.game_id.str.replace("_STL", "_LA").str.replace("_SD", "_LAC").str.replace("_OAK", "_LV")
        d["posteam"] = d.posteam.replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
        frames.append(team_game_stats(d))
    s = pd.concat(frames)
    # nflverse uses LA / LAC etc consistently across schedules and pbp
    for side in ["home", "away"]:
        t = s.add_prefix(f"{side}_").rename(columns={f"{side}_game_id": "game_id", f"{side}_team": f"{side}_team"})
        g = g.merge(t, on=["game_id", f"{side}_team"], how="left")
    g.to_parquet(os.path.join(RAW, "game_stats.parquet"))
    done = g[g.result.notna()]
    print("games:", len(g), "completed:", len(done), "missing pbp:", done.home_epa.isna().sum())


if __name__ == "__main__":
    main()
