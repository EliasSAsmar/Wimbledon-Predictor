#"Authorization": "f25sgom2I06gkt--LAOr0A" curl -H "apikey: YOUR_API_KEY" https://tennis.sportdevs.com/countries

#curl -H "apikey: f25sgom2I06gkt--LAOr0A" https://tennis.sportdevs.com/countries


import requests
from datetime import datetime, timedelta

# Step 1: Get tomorrow's date
tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

# Endpoint to fetch matches by date
matches_url = "https://api.sportdevs.com/tennis/matches-by-date?date=eq.2025-06-18"
headers = {
      "apikey": "f25sgom2I06gkt--LAOr0A",
    'Accept': 'application/json'
}

# Get matches
matches_response = requests.get(matches_url, headers=headers)
try:
    data = matches_response.json()
except Exception as e:
    print("⚠️ Failed to parse JSON:", e)
    exit()

if not data or not isinstance(data, list) or 'matches' not in data[0]:
    print(f"⚠️ No matches found for {tomorrow}. Response:\n{data}")
    exit()

match_list = data[0]['matches']
if not match_list:
    print(f"📭 No matches scheduled for {tomorrow}.")
    exit()

# Step 2: Loop through each match and fetch match winner odds
for match in match_list:
    match_id = match.get("match_id")
    home = match.get("home_team", {}).get("name", "Unknown")
    away = match.get("away_team", {}).get("name", "Unknown")

    print(f"\n🎾 {home} vs {away} (match_id: {match_id})")

    odds_url = f"https://tennis.sportdevs.com/odds/match-winner?match_id=eq.{match_id}&is_live=eq.false"
    odds_response = requests.get(odds_url, headers=headers)
    odds_data = odds_response.json()

    if odds_data and isinstance(odds_data, list):
        found_odds = False
        for period in odds_data[0].get("periods", []):
            for outcome in period.get("odds", []):
                found_odds = True
                print(f"  {outcome['name']}: {outcome['value']}")
        if not found_odds:
            print("  🛑 Odds not yet available.")
    else:
        print("  🛑 No odds returned.")
