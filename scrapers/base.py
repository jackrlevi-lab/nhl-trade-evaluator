"""
Base scraper — all source-specific scrapers inherit from this.
Handles rate limiting, retries, and response caching so we never
hammer source sites and don't re-fetch data we already have.
"""

import requests
import time
import json
import hashlib
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("NHL_CACHE_DIR", ".scraper_cache"))
CACHE_DIR.mkdir(exist_ok=True)


class RateLimiter:
    """Simple token bucket — respects source sites."""

    def __init__(self, calls_per_minute: int = 10):
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        wait_time = self.min_interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_call = time.time()


class BaseScraper(ABC):
    """
    Base class for all NHL data scrapers.

    Subclasses implement:
        - source_name: str
        - scrape(): fetches and returns parsed data
        - parse(): converts raw HTML/JSON to list of dicts
    """

    SOURCE_NAME: str = "base"
    CALLS_PER_MINUTE: int = 10
    CACHE_TTL_HOURS: int = 6          # how long before we re-fetch

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (compatible; NHLTradeEvaluator/1.0; "
                "research project; contact: your@email.com)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/json",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.rate_limiter = RateLimiter(self.CALLS_PER_MINUTE)

    def _cache_key(self, url: str, params: dict = None) -> str:
        raw = url + json.dumps(params or {}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{self.SOURCE_NAME}_{key}.json"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
        return age < timedelta(hours=self.CACHE_TTL_HOURS)

    def _read_cache(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def _write_cache(self, path: Path, data: dict):
        with open(path, "w") as f:
            json.dump(data, f)

    def get(self, url: str, params: dict = None, force_refresh: bool = False) -> requests.Response:
        """
        Fetches URL with caching and rate limiting.
        Returns a mock Response-like object from cache or live fetch.
        """
        key = self._cache_key(url, params)
        cache_path = self._cache_path(key)

        if not force_refresh and self._is_cache_valid(cache_path):
            logger.debug(f"Cache hit: {url}")
            cached = self._read_cache(cache_path)
            # Return a simple namespace so callers use .text/.json()
            return _CachedResponse(cached["text"], cached["status_code"])

        self.rate_limiter.wait()
        logger.info(f"Fetching: {url}")

        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                self._write_cache(cache_path, {
                    "text": resp.text,
                    "status_code": resp.status_code,
                })
                return resp
            except requests.HTTPError as e:
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    @abstractmethod
    def scrape(self, season: str, **kwargs) -> list[dict]:
        """
        Fetch and parse data for a given season.
        season format: "20232024"
        Returns list of dicts ready for DB insertion.
        """
        pass

    def scrape_range(self, start_year: int, end_year: int) -> list[dict]:
        """Convenience method to scrape multiple seasons."""
        all_data = []
        for year in range(start_year, end_year + 1):
            season = f"{year}{year + 1}"
            logger.info(f"Scraping {self.SOURCE_NAME} season {season}")
            try:
                data = self.scrape(season)
                all_data.extend(data)
            except Exception as e:
                logger.error(f"Failed {season}: {e}")
        return all_data


class _CachedResponse:
    """Mimics requests.Response for cached data."""
    def __init__(self, text: str, status_code: int):
        self.text = text
        self.status_code = status_code

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"Status {self.status_code}")
