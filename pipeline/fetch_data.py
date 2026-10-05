"""Download nflverse schedules (with closing lines) and play-by-play."""
import os, sys, urllib.request, datetime

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")
FIRST = 2015
GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"


def get(url, path):
    print("fetch", url)
    urllib.request.urlretrieve(url, path)


def main(current_season: int, refresh_all=False):
    os.makedirs(RAW, exist_ok=True)
    get(GAMES, os.path.join(RAW, "games.csv"))
    for y in range(FIRST, current_season + 1):
        p = os.path.join(RAW, f"pbp_{y}.parquet")
        # past seasons never change; only re-download the current one
        if refresh_all or y == current_season or not os.path.exists(p):
            get(PBP.format(y=y), p)


if __name__ == "__main__":
    today = datetime.date.today()
    season = today.year if today.month >= 8 else today.year - 1
    main(int(sys.argv[1]) if len(sys.argv) > 1 else season)
