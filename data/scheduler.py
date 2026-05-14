"""
Update scheduler — runs scrapers on a cadence and upserts results to DB.
Run this as a background process or cron job.

Usage:
    python -m nhl_trade_evaluator.data.scheduler          # runs continuously
    python -m nhl_trade_evaluator.data.scheduler --once   # single run and exit
    python -m nhl_trade_evaluator.data.scheduler --season 20232024  # specific season
"""

import schedule
import time
import logging
import argparse
from datetime import datetime
from sqlalchemy.dialects.sqlite import insert

from .database import init_db, get_session, PlayerSeason, PlayerWAR, PlayerContract
from ..scrapers.nst_scraper import NSTScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def current_season() -> str:
    """Returns current NHL season string e.g. '20232024'."""
    now = datetime.utcnow()
    # NHL season starts in October
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}{year + 1}"


def upsert_player_seasons(rows: list[dict], session):
    """Insert or update player season rows."""
    if not rows:
        return
    stmt = insert(PlayerSeason).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id", "season", "situation"],
        set_={
            col: stmt.excluded[col]
            for col in rows[0].keys()
            if col not in ("player_id", "season", "situation")
        }
    )
    session.execute(stmt)
    session.commit()
    logger.info(f"Upserted {len(rows)} player season rows")


def run_nst_update(season: str = None):
    season = season or current_season()
    logger.info(f"Running NST update for season {season}")

    scraper = NSTScraper()
    try:
        rows = scraper.scrape(season)
        session = get_session()
        upsert_player_seasons(rows, session)
        session.close()
        logger.info(f"NST update complete: {len(rows)} rows")
    except Exception as e:
        logger.error(f"NST update failed: {e}", exc_info=True)


def run_all_updates(season: str = None):
    """Run all scrapers in sequence."""
    run_nst_update(season)
    # run_evolving_hockey_update(season)   # add when built
    # run_puckpedia_update()               # contracts don't need season param


def setup_schedule():
    """
    During season (Oct–Apr): update stats daily at 6am UTC.
    Always: contracts weekly on Monday.
    """
    schedule.every().day.at("06:00").do(run_all_updates)
    schedule.every().monday.at("07:00").do(run_all_updates)
    logger.info("Schedule configured. Running...")


def main():
    parser = argparse.ArgumentParser(description="NHL data update scheduler")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--season", type=str, help="Specific season e.g. 20232024")
    args = parser.parse_args()

    init_db()

    if args.once or args.season:
        run_all_updates(season=args.season)
        return

    setup_schedule()
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
