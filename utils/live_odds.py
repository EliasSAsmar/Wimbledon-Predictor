import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.the-odds-api.com/v4/sports/tennis_atp_wimbledon/odds/"

params = {
    "apiKey": API_KEY,
    "regions": "us",                  # or "us,uk" if you want more coverage
    "markets": "h2h",
    "oddsFormat": "american",
    "dateFormat": "iso"
}

response = requests.get(BASE_URL, params=params)

if response.status_code != 200:
    print("Failed:", response.status_code, response.text)
else:
    data = response.json()
    for match in data:
        print("Match:", match["home_team"], "vs", match["away_team"])
        for book in match["bookmakers"]:
            print(f"  📗 {book['title']}:")
            for market in book["markets"]:
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        print(f"    {outcome['name']}: {outcome['price']}")

