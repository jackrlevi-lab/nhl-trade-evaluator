"""
Evolving Hockey scraper — WAR and GSVA estimates.
EH requires a session cookie from a logged-in account for full data access.
Free tier exposes enough for our validation purposes.

We target their skater exports:
  https://evolving-hockey.com/stats/skaters/

EH is more scraper-sensitive than NST. Strategy:
  1. Use session-based requests with realistic headers
  2. Conservative rate limiting (4 calls/min)
  3. Cache aggressively (24hr TTL — WAR doesn't change intra-day)
  4. Graceful fallback to manual CSV if blocked
"""

import csv
import io
import logging
from pathlib import Path
from .base import BaseScraper

logger = logging.getLogger(__name__)

EH_BASE = "https://evolving-hockey.com"
EH_SKATERS = f"{EH_BASE}/stats/skaters/"

COLUMN_MAP = {
    "Player":    "player_name",
    "Team":      "team",
    "Position":  "position",
    "GP":        "gp",
    "WAR":       "war",
    "GSVA":      "gsva",
    "Off WAR":   "off_war",
    "Def WAR":   "def_war",
}


class EvolvingHockeyScraper(BaseScraper):
    SOURCE_NAME = "eh"
    CALLS_PER_MINUTE = 4        # more conservative than NST
    CACHE_TTL_HOURS = 24        # WAR updates daily at most

    def scrape(self, season: str, **kwargs) -> list[dict]:
        """
        Attempt live scrape. Fall back to manual CSV if blocked.
        season: "20232024"
        """
        try:
            return self._scrape_live(season)
        except Exception as e:
            logger.warning(f"EH live scrape failed ({e}). Trying manual fallback.")
            return self._load_manual_csv(season)

    def _scrape_live(self, season: str) -> list[dict]:
        """
        EH exposes a hidden JSON endpoint their frontend uses.
        Format: /api/WAR/skaters?season={season}&sit=ev&rate=false
        This is more stable than scraping rendered HTML.
        """
        season_fmt = f"{season[:4]}-{season[4:]}"   # "20232024" -> "2023-2024"

        params = {
            "season":   season_fmt,
            "sit":      "ev",
            "score":    "all",
            "rate":     "false",
            "pos":      "all",
            "loc":      "all",
            "toi":      "0",
            "gpfilt":   "none",
        }

        resp = self.get(f"{EH_BASE}/api/WAR/skaters", params=params)
        data = resp.json()

        rows = []
        for player in data.get("data", []):
            row = {
                "season":      season,
                "player_name": player.get("Player", ""),
                "team":        player.get("Team", ""),
                "position":    player.get("Position", ""),
                "gp":          self._cast(player.get("GP")),
                "war":         self._cast(player.get("WAR")),
                "gsva":        self._cast(player.get("GSVA")),
                "off_war":     self._cast(player.get("Off WAR")),
                "def_war":     self._cast(player.get("Def WAR")),
            }
            if row["player_name"]:
                rows.append(row)

        logger.info(f"EH live scrape {season}: {len(rows)} players")
        return rows

    def _load_manual_csv(self, season: str) -> list[dict]:
        """
        Fallback: load from manually downloaded CSV.
        Place file at: data/manual_imports/eh_war_{season}.csv
        Download from: https://evolving-hockey.com/stats/skaters/
        """
        path = Path(f"data/manual_imports/eh_war_{season}.csv")

        if not path.exists():
            logger.error(
                f"Manual CSV not found at {path}. "
                f"Download from evolving-hockey.com and place it there."
            )
            return []

        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {"season": season}
                for src, dst in COLUMN_MAP.items():
                    parsed[dst] = self._cast(row.get(src, ""))
                if parsed.get("player_name"):
                    rows.append(parsed)

        logger.info(f"EH manual CSV {season}: {len(rows)} players")
        return rows

    def _cast(self, value):
        if value is None or str(value).strip() in ("", "-", "N/A"):
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            pass
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)
