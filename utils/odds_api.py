import requests
import logging

API_KEY = "8918f3758b083e5533c7d508b8957223"
SPORTS_KEY = "tennis_atp_wimbledon"

BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORTS_KEY}/odds/"

def fetch_wimbledon_odds(region='us', market='h2h', odds_format='american'):
    params = {
        'apiKey': API_KEY,
        'regions': region,
        'markets': market,
        'oddsFormat': odds_format,
        'dateFormat': "iso"
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        odds_data = response.json()
        return parse_odds_data(odds_data)
    except Exception as e:
        logging.error(f"Failed to fetch odds: {e}")
        return {}

def parse_odds_data(odds_data):
    structured = {}
    for match in odds_data:
        home = match["home_team"]
        away = match["away_team"]

        # Pull odds from DraftKings if available; fallback to first book
        book = next((b for b in match["bookmakers"] if b["key"] == "draftkings"), match["bookmakers"][0])
        market = next((m for m in book["markets"] if m["key"] == "h2h"), None)

        if market and len(market["outcomes"]) == 2:
            p1 = market["outcomes"][0]
            p2 = market["outcomes"][1]

            structured[(home, away)] = {
                "player1": {"name": p1["name"], "decimal": p1["price"]},
                "player2": {"name": p2["name"], "decimal": p2["price"]}
            }
    return structured