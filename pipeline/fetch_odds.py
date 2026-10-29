"""Pull current DraftKings NFL spreads and totals from The Odds API.

Needs env var ODDS_API_KEY (stored as a GitHub Actions secret, never in code).
One call uses 2 credits (spreads + totals). DraftKings and the books in
EXTRA_BOOKS are pulled together; the API bills each group of up to 10 books as
one region, so the extra books cost nothing extra (keep the total at 10 or less). Without a key this
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

# Extra books shown on the line movement chart: Odds API key -> short code used in the data
EXTRA_BOOKS = {"fanduel": "fd", "williamhill_us": "cz"}   # williamhill_us = Caesars


def fetch(key_env="ODDS_API_KEY", markets="spreads,totals"):
    """markets="spreads" costs 1 credit instead of 2 (used by the daytime line checks)."""
    key = os.environ.get(key_env)
    if not key:
        print(f"{key_env} not set; skipping DraftKings pull")
        return {}
    q = urllib.parse.urlencode({"apiKey": key, "regions": "us", "markets": markets,
                                "bookmakers": ",".join(["draftkings", *EXTRA_BOOKS]), "oddsFormat": "american"})
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
            if bk["key"] != "draftkings" and bk["key"] not in EXTRA_BOOKS:
                continue
            b = {}
            for m in bk["markets"]:
                if m["key"] == "spreads":
                    for o in m["outcomes"]:
                        if NAME_TO_ABBR.get(o["name"]) == h:
                            b["home_spread"] = o["point"]  # negative = home favored
                            b["home_price"] = o.get("price")
                        elif NAME_TO_ABBR.get(o["name"]) == a:
                            b["away_price"] = o.get("price")
                elif m["key"] == "totals":
                    b["total"] = m["outcomes"][0]["point"]
            if bk["key"] == "draftkings":   # DraftKings stays the main line, at the top level
                rec.update(b); rec["updated"] = bk.get("last_update")
            elif "home_spread" in b:
                rec[EXTRA_BOOKS[bk["key"]]] = b
        if "home_spread" in rec:
            out[f"{a}@{h}"] = rec
    print("DK lines pulled:", len(out), "| extra books:", {c: sum(1 for v in out.values() if c in v) for c in EXTRA_BOOKS.values()})
    return out


if __name__ == "__main__":
    print(json.dumps(fetch(), indent=1))
