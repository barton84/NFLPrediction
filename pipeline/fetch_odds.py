"""Pull current DraftKings NFL spreads and totals from The Odds API.

Needs env var ODDS_API_KEY (stored as a GitHub Actions secret, never in code).
One call uses 2 credits (spreads + totals, one region). Without a key this
script exits quietly and the site falls back to nflverse consensus lines.

Each pull is also saved to site/data/odds_history/<season>_<week>.json so next
week's "model vs actual" table can grade against the DK number we saw.
"""
import os, json, urllib.request, urllib.parse

ROOT = os.path.join(os.path.dirname(__file__), "..")
NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def fetch():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY not set; skipping DraftKings pull (consensus lines will be used)")
        return {}
    q = urllib.parse.urlencode({"apiKey": key, "regions": "us", "markets": "spreads,totals",
                                "bookmakers": "draftkings", "oddsFormat": "american"})
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?{q}"
    with urllib.request.urlopen(url, timeout=30) as r:
        print("odds api credits remaining:", r.headers.get("x-requests-remaining"))
        data = json.load(r)
    out = {}
    for ev in data:
        h, a = NAME_TO_ABBR.get(ev["home_team"]), NAME_TO_ABBR.get(ev["away_team"])
        if not h or not a:
            continue
        rec = {"commence": ev["commence_time"]}
        for bk in ev.get("bookmakers", []):
            if bk["key"] != "draftkings":
                continue
            for m in bk["markets"]:
                if m["key"] == "spreads":
                    for o in m["outcomes"]:
                        if NAME_TO_ABBR.get(o["name"]) == h:
                            rec["home_spread"] = o["point"]  # negative = home favored
                elif m["key"] == "totals":
                    rec["total"] = m["outcomes"][0]["point"]
            rec["updated"] = bk.get("last_update")
        if "home_spread" in rec:
            out[f"{a}@{h}"] = rec
    print("DK lines pulled:", len(out))
    return out


if __name__ == "__main__":
    print(json.dumps(fetch(), indent=1))
