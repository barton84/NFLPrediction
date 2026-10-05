"""Kickoff weather forecasts from Open-Meteo (free, no key).

Domes and retractable-roof stadiums are skipped. Forecasts reach about 16
days out, so this only runs for the current week's games. Any failure just
leaves that game without a forecast; the app then falls back to manual entry.
"""
import json, urllib.request, urllib.parse, datetime

# lat, lon, roof type ("outdoor", "dome", "retractable")
STADIUMS = {
    "ATL97": (33.755, -84.401, "retractable"), "BAL00": (39.278, -76.623, "outdoor"), "BOS00": (42.091, -71.264, "outdoor"),
    "BUF00": (42.774, -78.787, "outdoor"), "CAR00": (35.226, -80.853, "outdoor"), "CHI98": (41.862, -87.617, "outdoor"),
    "CIN00": (39.095, -84.516, "outdoor"), "CLE00": (41.506, -81.700, "outdoor"), "DAL00": (32.748, -97.093, "retractable"),
    "DEN00": (39.744, -105.020, "outdoor"), "DET00": (42.340, -83.046, "dome"), "GNB00": (44.501, -88.062, "outdoor"),
    "HOU00": (29.685, -95.411, "retractable"), "IND00": (39.760, -86.164, "retractable"), "JAX00": (30.324, -81.637, "outdoor"),
    "KAN00": (39.049, -94.484, "outdoor"), "LAX01": (33.953, -118.339, "dome"), "MIA00": (25.958, -80.239, "outdoor"),
    "MIN01": (44.974, -93.258, "dome"), "NAS00": (36.166, -86.771, "outdoor"), "NOR00": (29.951, -90.081, "dome"),
    "NYC01": (40.814, -74.074, "outdoor"), "PHI00": (39.901, -75.168, "outdoor"), "PHO00": (33.528, -112.263, "retractable"),
    "PIT00": (40.447, -80.016, "outdoor"), "SEA00": (47.595, -122.332, "outdoor"), "SFO01": (37.403, -121.970, "outdoor"),
    "TAM00": (27.976, -82.503, "outdoor"), "VEG00": (36.091, -115.184, "dome"), "WAS00": (38.908, -76.864, "outdoor"),
    "LON00": (51.556, -0.280, "outdoor"), "LON02": (51.604, -0.066, "outdoor"), "MAD01": (40.453, -3.688, "retractable"),
    "MEX00": (19.303, -99.150, "outdoor"), "MUN01": (48.219, 11.625, "outdoor"), "PAR00": (48.924, 2.360, "outdoor"),
    "RIO00": (-22.912, -43.230, "outdoor"), "MEL00": (-37.820, 144.983, "outdoor"),
}
# a home team's stadium_id can be reused for an international game, so the name wins when known
BY_NAME = {"Tottenham Hotspur Stadium": "LON02", "Wembley Stadium": "LON00", "Bernabeu": "MAD01",
           "Estadio Banorte": "MEX00", "Allianz Arena": "MUN01", "FC Bayern Munich Stadium": "MUN01",
           "Stade de France": "PAR00", "Maracana Stadium": "RIO00", "Melbourne Cricket Ground": "MEL00"}


def venue(stadium_id, stadium_name, roof):
    sid = BY_NAME.get(stadium_name, stadium_id)
    lat, lon, kind = STADIUMS.get(sid, (None, None, "outdoor"))
    if roof in ("dome", "closed"):
        kind = "dome" if kind != "retractable" else "retractable"
    return lat, lon, kind


def summarize(data, kick_hour):
    """Average the 3 hours from kickoff. data = Open-Meteo hourly JSON (ET timezone)."""
    h = data["hourly"]
    idx = [i for i, t in enumerate(h["time"]) if kick_hour <= int(t[11:13]) < kick_hour + 3] or [0]
    avg = lambda k: sum((h[k][i] or 0) for i in idx) / len(idx)
    mx = lambda k: max((h[k][i] or 0) for i in idx)
    return {"temp": round(avg("temperature_2m")), "wind": round(avg("wind_speed_10m")), "gust": round(mx("wind_gusts_10m")),
            "precip_prob": round(mx("precipitation_probability")), "precip_in": round(sum((h["precipitation"][i] or 0) for i in idx), 2),
            "snow_in": round(sum((h["snowfall"][i] or 0) for i in idx), 1)}


def forecast(game):
    """game: dict with stadium_id, stadium, roof, date (YYYY-MM-DD, ET), time (HH:MM ET)."""
    lat, lon, kind = venue(game.get("stadium_id"), game.get("stadium"), game.get("roof"))
    if kind in ("dome", "retractable"):
        return {"indoor": True, "kind": kind}
    if lat is None:
        return None
    day = datetime.date.fromisoformat(game["date"])
    if (day - datetime.date.today()).days > 15:
        return None
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "America/New_York",
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability,precipitation,snowfall",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
        "start_date": game["date"], "end_date": game["date"]})
    try:
        with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=20) as r:
            data = json.load(r)
        out = summarize(data, int((game.get("time") or "13:00")[:2]))
        out.update(indoor=False, kind="outdoor", fetched=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"))
        return out
    except Exception as e:
        print("weather fetch failed for", game.get("id"), e)
        return None
