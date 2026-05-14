"""
Database layer — SQLite for development, swap connection string for PostgreSQL in production.
All scraped data lands here. Frontend never touches source sites directly.
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, UniqueConstraint, Index, text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime
import os

DB_PATH = os.environ.get("NHL_DB_PATH", "nhl_trade_evaluator.db")
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Session = sessionmaker(bind=ENGINE)


class Base(DeclarativeBase):
    pass


class PlayerSeason(Base):
    """
    Core skater stats — one row per player per season per situation.
    Sourced from Natural Stat Trick.
    situation: 'all', 'ev', 'pp', 'pk'
    """
    __tablename__ = "player_seasons"

    id            = Column(Integer, primary_key=True)
    player_id     = Column(String, nullable=False)   # NST internal ID
    player_name   = Column(String, nullable=False)
    season        = Column(String, nullable=False)   # e.g. "20232024"
    team          = Column(String)
    position      = Column(String)
    situation     = Column(String, nullable=False)
    gp            = Column(Integer)
    toi           = Column(Float)                    # minutes
    goals         = Column(Integer)
    assists        = Column(Integer)
    points        = Column(Integer)
    ixg           = Column(Float)                    # individual expected goals
    icf           = Column(Float)                    # individual corsi for
    xgf_pct       = Column(Float)                    # expected goals for %
    xgf_per60     = Column(Float)
    xga_per60     = Column(Float)
    cf_pct        = Column(Float)                    # corsi for %
    rel_cf_pct    = Column(Float)                    # relative corsi
    ozs_pct       = Column(Float)                    # offensive zone start %
    scraped_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_id", "season", "situation", name="uq_player_season_sit"),
        Index("ix_player_season", "player_name", "season"),
    )


class PlayerWAR(Base):
    """
    WAR and GSVA estimates — sourced from Evolving Hockey.
    Used as validation benchmark against our own model output.
    """
    __tablename__ = "player_war"

    id          = Column(Integer, primary_key=True)
    player_name = Column(String, nullable=False)
    season      = Column(String, nullable=False)
    team        = Column(String)
    position    = Column(String)
    gp          = Column(Integer)
    war         = Column(Float)
    gsva        = Column(Float)
    off_war     = Column(Float)
    def_war     = Column(Float)
    scraped_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_name", "season", name="uq_war_player_season"),
    )


class PlayerContract(Base):
    """
    Contract data — sourced from PuckPedia.
    cap_hit is AAV. years_remaining counts from current season.
    """
    __tablename__ = "player_contracts"

    id              = Column(Integer, primary_key=True)
    player_name     = Column(String, nullable=False)
    team            = Column(String)
    cap_hit         = Column(Float)                  # AAV in dollars
    total_value     = Column(Float)
    years_remaining = Column(Integer)
    expiry_status   = Column(String)                 # UFA, RFA, ELC
    has_nmc         = Column(Integer, default=0)     # 1 = full NMC
    has_ntc         = Column(Integer, default=0)     # 1 = modified NTC
    scraped_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_name", name="uq_contract_player"),
    )


class DraftPick(Base):
    """
    Historical draft data for pick value model.
    nhl_gp and career_war populated once player has 3+ NHL seasons.
    """
    __tablename__ = "draft_picks"

    id            = Column(Integer, primary_key=True)
    year          = Column(Integer, nullable=False)
    round         = Column(Integer)
    overall       = Column(Integer)
    player_name   = Column(String)
    team          = Column(String)
    nhl_gp        = Column(Integer, default=0)
    career_war    = Column(Float, default=0.0)
    reached_nhl   = Column(Integer, default=0)       # 1 = 100+ GP
    scraped_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("year", "overall", name="uq_draft_year_pick"),
    )


class PlayerValuation(Base):
    """
    Our model's output — computed, not scraped.
    Recomputed whenever underlying data updates.
    """
    __tablename__ = "player_valuations"

    id                  = Column(Integer, primary_key=True)
    player_name         = Column(String, nullable=False)
    season              = Column(String, nullable=False)
    team                = Column(String)
    position            = Column(String)
    age                 = Column(Integer)
    war_estimate        = Column(Float)              # our model
    contract_efficiency = Column(Float)              # WAR per $1M cap
    trade_value         = Column(Float)              # composite score
    peak_war_proj       = Column(Float)              # age curve projection
    computed_at         = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_name", "season", name="uq_val_player_season"),
    )


def init_db():
    Base.metadata.create_all(ENGINE)
    print(f"Database initialized at {DB_PATH}")


def get_session():
    return Session()


if __name__ == "__main__":
    init_db()
