from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import os

# Create base class for declarative models
Base = declarative_base()

# Database connection
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///database/tennis.db')
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

class Prediction(Base):
    """Model for storing match predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    player1 = Column(String, nullable=False)
    player2 = Column(String, nullable=False)
    player1_win_prob = Column(Float, nullable=False)
    player2_win_prob = Column(Float, nullable=False)
    prediction_date = Column(DateTime, default=datetime.now)
    match_date = Column(Date)
    tournament = Column(String)
    round = Column(String)
    model_used = Column(String, default='RFSR_ensemble')
    
    # Relationships
    betting_odds = relationship("BettingOdds", back_populates="prediction")
    ev_plays = relationship("EVPlay", back_populates="prediction")
    results = relationship("Result", back_populates="prediction")
    
    def __repr__(self):
        return f"<Prediction({self.player1} vs {self.player2}, {self.prediction_date})>"


class BettingOdds(Base):
    """Model for storing sportsbook odds"""
    __tablename__ = 'betting_odds'
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=False)
    sportsbook = Column(String, nullable=False)
    player1_odds = Column(Integer, nullable=False)  # American odds format
    player2_odds = Column(Integer, nullable=False)  # American odds format
    player1_implied_prob = Column(Float, nullable=False)
    player2_implied_prob = Column(Float, nullable=False)
    odds_date = Column(DateTime, default=datetime.now)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="betting_odds")
    ev_plays = relationship("EVPlay", back_populates="betting_odds")
    
    def __repr__(self):
        return f"<BettingOdds({self.sportsbook}, {self.odds_date})>"


class EVPlay(Base):
    """Model for storing qualifying EV plays"""
    __tablename__ = 'ev_plays'
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=False)
    betting_odds_id = Column(Integer, ForeignKey('betting_odds.id'), nullable=False)
    player1_ev = Column(Float, nullable=False)
    player2_ev = Column(Float, nullable=False)
    threshold_met = Column(Boolean, default=False)
    status = Column(String, default='active')  # active, completed, cancelled
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="ev_plays")
    betting_odds = relationship("BettingOdds", back_populates="ev_plays")
    results = relationship("Result", back_populates="ev_play")
    
    def __repr__(self):
        return f"<EVPlay({self.prediction.player1} vs {self.prediction.player2}, {self.status})>"


class Result(Base):
    """Model for storing match outcomes"""
    __tablename__ = 'results'
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=False)
    ev_play_id = Column(Integer, ForeignKey('ev_plays.id'), nullable=False)
    winner = Column(String, nullable=False)  # player1 or player2
    profit_loss = Column(Float)  # NULL if not bet on
    settlement_date = Column(DateTime, default=datetime.now)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="results")
    ev_play = relationship("EVPlay", back_populates="results")
    
    def __repr__(self):
        return f"<Result({self.winner}, {self.settlement_date})>"


# Create all tables
def init_db():
    Base.metadata.create_all(engine)


# Helper function to get a database session
def get_session():
    return Session() 