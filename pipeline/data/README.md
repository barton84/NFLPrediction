# win_totals.csv

Each team's regular-season win total from just before Week 1. The model uses it as the starting rating for each season (model v2.1+).

- 2015-2025: Covers / SportsOddsHistory, "NFL regular season win total results by team"
  https://www.covers.com/sportsoddshistory/nfl-regular-season-win-total-results-by-team/
  https://www.covers.com/sportsoddshistory/nfl-regular-season-win-total-results-by-team-2010s/
- 2026: Sharp Football Analysis, published 2026-09-02
  https://www.sharpfootballanalysis.com/betting/nfl-team-win-totals-odds-best-bets/

**Once a year, before Week 1:** add 32 rows for the new season (team,season,line). Use full team names as in the
existing rows. Without them the model falls back to starting each team at half of last season's rating.
