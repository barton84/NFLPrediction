# NFL Spread Model

A weekly NFL point-spread model with a static web front end. It compares the model's numbers to DraftKings lines, flags games where the two disagree by 5+ points, and grades last week's projections against the actual results.

## How it works

```
nflverse (schedules, closing lines, play-by-play)  ─┐
The Odds API (DraftKings spreads and totals)        ─┤─> pipeline/run.py ─> site/data/model.json ─> site/index.html
```

- `pipeline/fetch_data.py` downloads nflverse schedules and play-by-play (2015 to now).
- `pipeline/build_games.py` builds per-game stats: EPA/play, success rate, yards/play, with garbage-time plays removed.
- `pipeline/ratings.py` computes point-in-time team ratings. They are opponent-adjusted, blended with last season early in the year, and use only games played before each week.
- `pipeline/qb.py` rates every QB by career EPA per dropback and measures each game's QB change against the team's recent starters.
- `pipeline/backtest.py` runs the leave-one-season-out testing.
- `pipeline/fetch_odds.py` pulls DraftKings lines. It needs `ODDS_API_KEY`. Without the key, the site falls back to consensus lines.
- `pipeline/run.py` runs everything and writes `site/data/model.json`.
- `site/app.html` is the page. `pipeline/make_index.py` wraps it into `site/index.html`.

## Run locally

```bash
pip install -r requirements.txt
export ODDS_API_KEY=your_key_here     # optional
python pipeline/run.py                # first run downloads about 220 MB
python pipeline/make_index.py
cd site && python -m http.server 8000 # open http://localhost:8000
```

## Deploy (GitHub + Netlify)

1. Create a GitHub repo and push this folder.
2. In the repo, go to Settings > Secrets and variables > Actions > New repository secret. Name it `ODDS_API_KEY` and paste in your key.
3. In the Actions tab, open "Update model data" and click "Run workflow" once to confirm it works.
4. In Netlify, choose Add new site > Import from Git and pick the repo. Leave the build command blank and set the publish directory to `site`. The `netlify.toml` file already sets this.

The workflow runs every morning, commits the refreshed data, and Netlify redeploys automatically. Each run uses 2 Odds API credits, so about 60 a month.

## Current tested weights (2016-2025, 2,761 games)

| Signal | Weight | Holdout range |
|---|---|---|
| Home field | 1.86 pts | 1.72 to 2.07 |
| Scoring margin rating | 0.73 | 0.70 to 0.78 |
| Success rate rating | 36 per 1.0 net SR | 32 to 41 |
| QB change (EPA/dropback) | 25.1 | 22.7 to 28.0 |
| Rest days | 0.15 per day | 0.12 to 0.19 |

The model's error is 12.99 RMSE, against 12.70 for the closing line. Against the spread it lands near 49% overall. Treat flags as a reason to look closer, not as bets.
