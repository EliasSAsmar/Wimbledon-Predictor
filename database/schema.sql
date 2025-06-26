-- Tennis Prediction Database Schema

-- Predictions table: Store model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player1 TEXT NOT NULL,
    player2 TEXT NOT NULL,
    player1_win_prob REAL NOT NULL,
    player2_win_prob REAL NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    match_date DATE,
    tournament TEXT,
    round TEXT,
    model_used TEXT DEFAULT 'RFSR_ensemble'
);

-- Betting odds table: Store sportsbook odds
CREATE TABLE IF NOT EXISTS betting_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    sportsbook TEXT NOT NULL,
    player1_odds INTEGER NOT NULL,  -- American odds format
    player2_odds INTEGER NOT NULL,  -- American odds format
    player1_implied_prob REAL NOT NULL,
    player2_implied_prob REAL NOT NULL,
    odds_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

-- EV plays table: Store qualifying plays above threshold
CREATE TABLE IF NOT EXISTS ev_plays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    betting_odds_id INTEGER NOT NULL,
    player1_ev REAL NOT NULL,  -- Expected value for player 1
    player2_ev REAL NOT NULL,  -- Expected value for player 2
    threshold_met BOOLEAN DEFAULT FALSE,
    status TEXT DEFAULT 'active',  -- active, completed, cancelled
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id),
    FOREIGN KEY (betting_odds_id) REFERENCES betting_odds(id)
);

-- Results table: Store match outcomes
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    ev_play_id INTEGER NOT NULL,
    winner TEXT NOT NULL,  -- player1 or player2
    profit_loss REAL,  -- NULL if not bet on
    settlement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id),
    FOREIGN KEY (ev_play_id) REFERENCES ev_plays(id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_ev_plays_status ON ev_plays(status);
CREATE INDEX IF NOT EXISTS idx_results_settlement ON results(settlement_date); 