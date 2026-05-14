"""
Hockey Reference draft scraper — historical draft data 2000-present.
Used to build the pick value curve.

URL pattern:
  https://www.hockey-reference.com/draft/NHL_{year}_entry.html

We extract: year, round, overall pick, player name, team, GP, and
points to later join with career WAR data for pick value modeling.

Hockey Reference is generally scraper-friendly but has rate limits.
We use conservative pacing and aggressive caching.
"""

import logging
import re
from bs4 import BeautifulSoup
from .base import BaseScraper

logger = logging.getLogger(__name__)

HOCKEYREF_BASE = "https://www.hockey-reference.com"
NHL_GP_THRESHOLD = 100          # minimum GP to count as "reached NHL"


class HockeyRefDraftScraper(BaseScraper):
    SOURCE_NAME = "hockeyref"
    CALLS_PER_MINUTE = 8
    CACHE_TTL_HOURS = 168       # draft data is historical — cache for 1 week

    def scrape(self, season: str, **kwargs) -> list[dict]:
        """
        season: "20002001" — we extract the draft year (2000).
        Returns list of dicts matching DraftPick schema.
        """
        year = int(season[:4])
        return self._scrape_draft_year(year)

    def scrape_range(self, start_year: int, end_year: int) -> list[dict]:
        """Scrape multiple draft years. Use 2000-2020 for pick value model."""
        all_picks = []
        for year in range(start_year, end_year + 1):
            logger.info(f"Scraping {year} NHL draft")
            try:
                picks = self._scrape_draft_year(year)
                all_picks.extend(picks)
                logger.info(f"  {year}: {len(picks)} picks")
            except Exception as e:
                logger.error(f"  {year} failed: {e}")
        return all_picks

    def _scrape_draft_year(self, year: int) -> list[dict]:
        url = f"{HOCKEYREF_BASE}/draft/NHL_{year}_entry.html"
        resp = self.get(url)
        return self._parse_draft_table(resp.text, year)

    def _parse_draft_table(self, html: str, year: int) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": "picks"})

        if not table:
            logger.warning(f"No picks table found for {year}")
            return []

        picks = []
        tbody = table.find("tbody")
        if not tbody:
            return []

        for row in tbody.find_all("tr"):
            # Skip header rows that repeat mid-table
            if row.get("class") and "thead" in row.get("class"):
                continue

            cells = row.find_all(["td", "th"])
            if len(cells) < 6:
                continue

            try:
                pick = self._parse_row(cells, year)
                if pick:
                    picks.append(pick)
            except Exception as e:
                logger.debug(f"Row parse error {year}: {e}")

        return picks

    def _parse_row(self, cells: list, year: int) -> dict | None:
        """
        HockeyRef draft table columns (approximate):
        0: round, 1: overall, 2: team, 3: player, 4: nationality,
        5: position, 6: age, 7: GP, 8: G, 9: A, 10: Pts, ...
        """
        def text(cell):
            return cell.get_text(strip=True)

        def safe_int(cell):
            t = text(cell)
            try:
                return int(re.sub(r"[^\d]", "", t))
            except ValueError:
                return 0

        round_num = safe_int(cells[0])
        overall   = safe_int(cells[1])
        team      = text(cells[2])
        player    = text(cells[3])

        # Skip rows with no player (unfilled picks, etc.)
        if not player or player in ("", "Player"):
            return None

        position  = text(cells[5]) if len(cells) > 5 else ""
        nhl_gp    = safe_int(cells[7]) if len(cells) > 7 else 0

        return {
            "year":        year,
            "round":       round_num,
            "overall":     overall,
            "player_name": player,
            "team":        team,
            "position":    position,
            "nhl_gp":      nhl_gp,
            "reached_nhl": 1 if nhl_gp >= NHL_GP_THRESHOLD else 0,
            "career_war":  0.0,     # populated separately from EH WAR data
        }
