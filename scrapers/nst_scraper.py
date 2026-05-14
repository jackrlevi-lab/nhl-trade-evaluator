"""
Natural Stat Trick scraper — skater stats by season and situation.
NST uses a report URL pattern with CSV export endpoints.
We target the CSV export directly — cleaner than parsing HTML tables.

URL pattern:
  https://www.naturalstattrick.com/playerteams.php
  ?fromseason={season}&thruseason={season}&stype=2&sit={sit}&score=all
  &stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=GP&fd=&td=
  &tgp=410&lines=single&draftteam=ALL&format=csv
"""

import csv
import io
import logging
from .base import BaseScraper

logger = logging.getLogger(__name__)

NST_BASE = "https://www.naturalstattrick.com/playerteams.php"

SITUATION_MAP = {
    "all": "all",
    "ev":  "ev",
    "pp":  "pp",
    "pk":  "pk",
}

# Maps NST CSV column names to our DB column names
COLUMN_MAP = {
    "Player":      "player_name",
    "Team":        "team",
    "Position":    "position",
    "GP":          "gp",
    "TOI":         "toi",
    "Goals":       "goals",
    "Total Assists": "assists",
    "Total Points":  "points",
    "ixG":         "ixg",
    "iCF":         "icf",
    "xGF%":        "xgf_pct",
    "xGF/60":      "xgf_per60",
    "xGA/60":      "xga_per60",
    "CF%":         "cf_pct",
    "CF% Rel":     "rel_cf_pct",
    "Off. Zone Starts%": "ozs_pct",
}


class NSTScraper(BaseScraper):
    SOURCE_NAME = "nst"
    CALLS_PER_MINUTE = 6        # be respectful — NST is a small site

    def scrape(self, season: str, situations: list[str] = None) -> list[dict]:
        """
        Scrape all situations for a given season.
        season: "20232024"
        Returns list of dicts matching PlayerSeason schema.
        """
        situations = situations or list(SITUATION_MAP.keys())
        all_rows = []

        for sit in situations:
            rows = self._scrape_situation(season, sit)
            all_rows.extend(rows)
            logger.info(f"NST {season} {sit}: {len(rows)} players")

        return all_rows

    def _scrape_situation(self, season: str, situation: str) -> list[dict]:
        params = {
            "fromseason": season,
            "thruseason": season,
            "stype": "2",           # regular season
            "sit": SITUATION_MAP[situation],
            "score": "all",
            "stdoi": "std",
            "rate": "n",
            "team": "ALL",
            "pos": "S",             # skaters only
            "loc": "B",
            "toi": "0",
            "gpfilt": "GP",
            "tgp": "410",
            "lines": "single",
            "draftteam": "ALL",
            "format": "csv",
        }

        resp = self.get(NST_BASE, params=params)
        return self._parse_csv(resp.text, season, situation)

    def _parse_csv(self, text: str, season: str, situation: str) -> list[dict]:
        rows = []

        # NST sometimes prepends a blank line — strip it
        cleaned = text.strip()
        reader = csv.DictReader(io.StringIO(cleaned))

        for row in reader:
            parsed = {
                "season":    season,
                "situation": situation,
                "player_id": self._make_player_id(row.get("Player", ""), row.get("Team", "")),
            }

            for src_col, dst_col in COLUMN_MAP.items():
                raw = row.get(src_col, "").strip()
                parsed[dst_col] = self._cast(raw)

            # Only keep rows with meaningful TOI
            if parsed.get("toi") and float(parsed["toi"] or 0) > 0:
                rows.append(parsed)

        return rows

    def _make_player_id(self, name: str, team: str) -> str:
        """Stable ID from name — crude but sufficient for matching."""
        return name.lower().replace(" ", "_").replace("'", "")

    def _cast(self, value: str):
        """Try int → float → string."""
        if value in ("", "-", "N/A"):
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value
