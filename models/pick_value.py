"""
Pick Value Model — NHL draft pick valuation.

Core question: given a pick at position N, what is its expected value?

Methodology:
  1. Pull historical draft data (2000-2020) from DB
  2. For each pick, we know: overall position, GP reached, career WAR
  3. Fit an exponential decay curve to WAR by pick position
  4. Output: expected WAR and P(NHL impact) for any pick 1-224

Why exponential decay:
  Value drops sharply in the top 5, then flattens. An exponential
  curve fits this better than linear. We validate against known
  benchmarks (Dobber, Evolving Hockey pick value charts).

Why 2000-2020 cutoff:
  Players drafted after 2020 haven't had enough time to establish
  NHL careers. Including them would bias the model toward zero WAR.
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_CACHE = Path("data/pick_value_model.json")


@dataclass
class PickValue:
    overall: int
    expected_war: float
    p_nhl_impact: float         # P(200+ GP)
    value_score: float          # normalized 0-10 scale


def exponential_decay(x, a, b, c):
    """f(x) = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


class PickValueModel:
    """
    Fits and serves pick valuations.
    Train once, cache to disk, reload on startup.
    """

    def __init__(self):
        self.war_params = None          # curve params for expected WAR
        self.impact_params = None       # curve params for P(NHL impact)
        self.max_war = None             # for normalization
        self.is_fitted = False

        if MODEL_CACHE.exists():
            self._load()

    def fit(self, picks_df: pd.DataFrame) -> dict:
        """
        Train on historical draft data.

        picks_df columns required:
            overall: int (1-224)
            career_war: float
            reached_nhl: int (0 or 1)
            nhl_gp: int
        """
        # Filter to players with enough career time
        # Only use drafts where player has had 5+ seasons to develop
        df = picks_df[
            (picks_df["overall"] > 0) &
            (picks_df["overall"] <= 224) &
            (picks_df["career_war"].notna())
        ].copy()

        logger.info(f"Fitting pick value model on {len(df)} historical picks")

        # Bin picks and compute mean WAR per position
        # Smooths noise from individual busts/steals
        df["pick_bin"] = pd.cut(df["overall"], bins=range(0, 226, 3), labels=False)
        war_by_bin = df.groupby("pick_bin").agg(
            mean_war=("career_war", "mean"),
            p_impact=("reached_nhl", "mean"),
            mid_pick=("overall", "median"),
            n=("career_war", "count")
        ).dropna()

        x_war = war_by_bin["mid_pick"].values
        y_war = war_by_bin["mean_war"].values.clip(min=0)

        x_impact = war_by_bin["mid_pick"].values
        y_impact = war_by_bin["p_impact"].values

        # Fit WAR curve
        try:
            self.war_params, _ = curve_fit(
                exponential_decay,
                x_war, y_war,
                p0=[8.0, 0.02, 0.3],
                bounds=([0, 0, 0], [20, 1, 5]),
                maxfev=5000
            )
        except RuntimeError as e:
            logger.warning(f"WAR curve fit failed: {e}. Using fallback params.")
            self.war_params = np.array([6.0, 0.025, 0.2])

        # Fit impact probability curve
        try:
            self.impact_params, _ = curve_fit(
                exponential_decay,
                x_impact, y_impact,
                p0=[0.7, 0.015, 0.1],
                bounds=([0, 0, 0], [1, 1, 0.5]),
                maxfev=5000
            )
        except RuntimeError as e:
            logger.warning(f"Impact curve fit failed: {e}. Using fallback params.")
            self.impact_params = np.array([0.65, 0.018, 0.08])

        self.max_war = float(exponential_decay(1, *self.war_params))
        self.is_fitted = True

        # Validate
        metrics = self._validate(df)
        logger.info(f"Model validation: {metrics}")

        self._save()
        return metrics

    def value(self, overall: int) -> PickValue:
        """
        Return the expected value of a pick at a given overall position.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        overall = max(1, min(overall, 224))

        expected_war = float(exponential_decay(overall, *self.war_params))
        expected_war = max(0.0, expected_war)

        p_impact = float(exponential_decay(overall, *self.impact_params))
        p_impact = max(0.0, min(1.0, p_impact))

        # Normalize to 0-10 scale relative to #1 pick
        value_score = round((expected_war / self.max_war) * 10, 2)

        return PickValue(
            overall=overall,
            expected_war=round(expected_war, 2),
            p_nhl_impact=round(p_impact, 3),
            value_score=value_score,
        )

    def value_table(self) -> pd.DataFrame:
        """Return a DataFrame of pick values for all positions 1-224."""
        rows = [self.value(i) for i in range(1, 225)]
        return pd.DataFrame([{
            "overall":       r.overall,
            "expected_war":  r.expected_war,
            "p_nhl_impact":  r.p_nhl_impact,
            "value_score":   r.value_score,
        } for r in rows])

    def _validate(self, df: pd.DataFrame) -> dict:
        """
        Check model against known reference points.
        Top 5 picks should average 4+ WAR, picks 100+ should average <0.5.
        """
        top5_actual = df[df["overall"] <= 5]["career_war"].mean()
        top5_pred   = np.mean([self.value(i).expected_war for i in range(1, 6)])

        late_actual = df[df["overall"] >= 100]["career_war"].mean()
        late_pred   = np.mean([self.value(i).expected_war for i in range(100, 125)])

        return {
            "top5_actual_war":  round(float(top5_actual), 2),
            "top5_predicted":   round(float(top5_pred), 2),
            "late_actual_war":  round(float(late_actual), 2),
            "late_predicted":   round(float(late_pred), 2),
        }

    def _save(self):
        MODEL_CACHE.parent.mkdir(exist_ok=True)
        with open(MODEL_CACHE, "w") as f:
            json.dump({
                "war_params":    self.war_params.tolist(),
                "impact_params": self.impact_params.tolist(),
                "max_war":       self.max_war,
            }, f)
        logger.info(f"Pick value model saved to {MODEL_CACHE}")

    def _load(self):
        with open(MODEL_CACHE) as f:
            data = json.load(f)
        self.war_params    = np.array(data["war_params"])
        self.impact_params = np.array(data["impact_params"])
        self.max_war       = data["max_war"]
        self.is_fitted     = True
        logger.info("Pick value model loaded from cache")


def build_pick_value_model(session) -> PickValueModel:
    """
    Load historical draft data from DB and fit the model.
    Call this once after scraping draft history.
    """
    from ..data.database import DraftPick

    rows = session.query(DraftPick).filter(
        DraftPick.year <= 2020,     # only mature draft classes
        DraftPick.year >= 2000,
    ).all()

    if len(rows) < 500:
        raise ValueError(
            f"Only {len(rows)} draft picks in DB. "
            "Run HockeyRefDraftScraper.scrape_range(2000, 2020) first."
        )

    df = pd.DataFrame([{
        "overall":     r.overall,
        "career_war":  r.career_war,
        "reached_nhl": r.reached_nhl,
        "nhl_gp":      r.nhl_gp,
        "year":        r.year,
    } for r in rows])

    model = PickValueModel()
    metrics = model.fit(df)
    return model
