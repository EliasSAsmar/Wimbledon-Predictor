#!/usr/bin/env python3
"""
Test script for Wimbledon odds API
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from odds_api import fetch_wimbledon_odds

def test_odds_api():
    print("Testing Wimbledon Odds API...")
    print("=" * 40)
    
    try:
        odds_data = fetch_wimbledon_odds()
        
        if odds_data:
            print(f"✅ API is working! Found {len(odds_data)} matches with odds:")
            print()
            
            for match, odds in odds_data.items():
                player1, player2 = match
                print(f"Match: {player1} vs {player2}")
                print(f"  {player1}: {odds['player1']['decimal']}")
                print(f"  {player2}: {odds['player2']['decimal']}")
                print()
        else:
            print("⚠️  API returned no data - this might mean:")
            print("   - No matches are currently available")
            print("   - API key might need to be updated")
            print("   - Wimbledon hasn't started yet")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_odds_api() 