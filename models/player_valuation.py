"""
Player Valuation Model — the core engine of the trade evaluator.

Produces three outputs per player:
  1. war_estimate     — our WAR estimate combining NST metrics
  2. contract_efficiency — WAR per $1M cap hit
  3. trade_value      — composite score (0-10) suitable for trade comparison

Methodology:
  We do NOT replicate Evolving Hockey's full WAR model (that's a PhD project).
  Instead we build a regression model that predicts WAR from NST metrics,
  trained on seasons where we have both NST data and EH WAR.
  This gives us a defensible, explainable model we can improve over time.

Age curve:
  Uses comparables clustering — find the N most similar players at the
  same age, see how their WAR evolved, project forward.
  Simple but transparent and defensible in an interview.
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)

MODEL_CACHE = Path("data/player_valuation_model.json")

# Age curve peak by position — based on public research
# Forwards peak ~24-26, defensemen ~25-27
POSITION_PEAK_AGE = {
    "C":  25.5,
    "L":  25.0,
    "R":  25.0,
    "D":  26.5,
    "G":  27.0,
}

# Cap ceiling for contract efficiency normalization
NHL_CAP_2024 = 88_000_000


@dataclass
class PlayerValuationResult:
    player_name:         str
    season:              str
    team:                str
    position:            str
    age:                 int
    war_estimate:        float
    contract_efficiency: float     # WAR per $1M cap
    trade_value:         float     # 0-10 composite
    peak_war_proj:       float     # projected WAR at peak age
    age_curve_stage:     str       # "ascending", "peak", "declining"
    contract_note:       str       # "ELC", "UFA", "team_friendly", "overpaid"


class PlayerValuationModel:
    """
    Regression model predicting WAR from NST metrics.
    Trained on matched NST + EH seasons.
    """

    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self.is_fitted = False
        self.cv_r2 = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    FEATURES = [
        "xgf_per60",        # offensive contribution
        "xga_per60",        # defensive contribution (inverted)
        "rel_cf_pct",       # relative possession impact
        "ixg",              # individual shot quality
        "ozs_pct",          # deployment context
        "toi",              # ice time (proxy for coach trust)
    ]

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from EV situation stats.
        We use EV only — most predictive of true player value.
        """
        ev = df[df["situation"] == "ev"].copy()

        # xGA per 60 should be inverted — lower is better defensively
        ev["xga_per60_inv"] = ev["xga_per60"] * -1

        features = ev[[
            "player_name", "season",
            "xgf_per60", "xga_per60_inv",
            "rel_cf_pct", "ixg",
            "ozs_pct", "toi"
        ]].dropna()

        return features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, nst_df: pd.DataFrame, war_df: pd.DataFrame) -> dict:
        """
        Train on matched NST + EH data.

        nst_df: from PlayerSeason table (all situations)
        war_df: from PlayerWAR table
        """
        features_df = self._build_features(nst_df)

        # Match on player + season
        merged = features_df.merge(
            war_df[["player_name", "season", "war"]],
            on=["player_name", "season"],
            how="inner"
        ).dropna(subset=["war"])

        logger.info(f"Training on {len(merged)} player-seasons")

        feature_cols = [c for c in merged.columns
                        if c not in ("player_name", "season", "war")]
        self.feature_names = feature_cols

        X = merged[feature_cols].values
        y = merged["war"].values

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ])

        # Cross-validate
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="r2")
        self.cv_r2 = float(cv_scores.mean())
        logger.info(f"Cross-val R²: {self.cv_r2:.3f}")

        self.pipeline.fit(X, y)
        self.is_fitted = True

        train_pred = self.pipeline.predict(X)
        train_r2   = r2_score(y, train_pred)

        metrics = {
            "n_samples":  len(merged),
            "cv_r2_mean": round(self.cv_r2, 3),
            "train_r2":   round(train_r2, 3),
        }
        logger.info(f"Model metrics: {metrics}")
        self._save()
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def value_player(
        self,
        player_name: str,
        season: str,
        nst_row: dict,
        contract: dict = None,
        age: int = None,
        position: str = None,
    ) -> PlayerValuationResult:
        """
        Compute trade value for a single player.

        nst_row: dict of EV stats for this player/season
        contract: dict with cap_hit, years_remaining, expiry_status
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        # WAR estimate
        x = np.array([[
            nst_row.get("xgf_per60", 0),
            -(nst_row.get("xga_per60", 0)),
            nst_row.get("rel_cf_pct", 0),
            nst_row.get("ixg", 0),
            nst_row.get("ozs_pct", 50),
            nst_row.get("toi", 0),
        ]])
        war_est = float(self.pipeline.predict(x)[0])
        war_est = max(-3.0, min(war_est, 10.0))     # clip extremes

        # Contract efficiency
        cap_hit = contract.get("cap_hit", 0) if contract else 0
        cap_pct = cap_hit / NHL_CAP_2024
        efficiency = (war_est / (cap_hit / 1_000_000)) if cap_hit > 0 else 0.0

        # Age curve projection
        peak_war, age_stage = self._project_peak(war_est, age, position)

        # Contract note
        contract_note = self._classify_contract(
            war_est, cap_hit, contract, efficiency
        )

        # Composite trade value (0-10)
        trade_value = self._compute_trade_value(
            war_est, efficiency, peak_war, age,
            contract.get("years_remaining", 0) if contract else 0,
            contract.get("has_nmc", 0) if contract else 0,
        )

        return PlayerValuationResult(
            player_name=player_name,
            season=season,
            team=nst_row.get("team", ""),
            position=position or nst_row.get("position", ""),
            age=age or 0,
            war_estimate=round(war_est, 2),
            contract_efficiency=round(efficiency, 2),
            trade_value=round(trade_value, 2),
            peak_war_proj=round(peak_war, 2),
            age_curve_stage=age_stage,
            contract_note=contract_note,
        )

    def _project_peak(
        self, war_est: float, age: int, position: str
    ) -> tuple[float, str]:
        """
        Simple age curve projection.
        Assumes players improve until peak age, then decline ~0.2 WAR/yr.
        """
        if not age or not position:
            return war_est, "unknown"

        pos = position[0].upper() if position else "C"
        peak_age = POSITION_PEAK_AGE.get(pos, 25.5)

        years_to_peak = peak_age - age

        if years_to_peak > 1.5:
            stage = "ascending"
            # Modest improvement projection — don't overfit
            peak_war = war_est + (years_to_peak * 0.25)
        elif years_to_peak >= -1.5:
            stage = "peak"
            peak_war = war_est * 1.05
        else:
            stage = "declining"
            years_past_peak = abs(years_to_peak)
            peak_war = war_est + (years_past_peak * 0.2)   # back-project peak

        return max(0.0, peak_war), stage

    def _classify_contract(
        self, war: float, cap_hit: float, contract: dict, efficiency: float
    ) -> str:
        if not contract:
            return "unknown"
        status = contract.get("expiry_status", "")
        if status == "ELC":
            return "ELC"
        if efficiency > 0.8:
            return "team_friendly"
        if efficiency < 0.3 and war > 0:
            return "overpaid"
        if status == "UFA":
            return "UFA"
        return "standard"

    def _compute_trade_value(
        self, war: float, efficiency: float, peak_war: float,
        age: int, years_remaining: int, has_nmc: int
    ) -> float:
        """
        Composite trade value on 0-10 scale.

        Components:
          - Current WAR (40%)
          - Contract efficiency (25%)
          - Future value / peak projection (25%)
          - Control years (10%)
          - NMC penalty (-0.5 if present)
        """
        # Normalize each component to 0-10
        war_score        = min(10, max(0, (war / 6.0) * 10))
        efficiency_score = min(10, max(0, efficiency * 5))
        future_score     = min(10, max(0, (peak_war / 6.0) * 10))
        control_score    = min(10, (years_remaining / 8.0) * 10)

        composite = (
            war_score        * 0.40 +
            efficiency_score * 0.25 +
            future_score     * 0.25 +
            control_score    * 0.10
        )

        if has_nmc:
            composite = max(0, composite - 0.5)

        return round(composite, 2)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        MODEL_CACHE.parent.mkdir(exist_ok=True)
        ridge = self.pipeline.named_steps["ridge"]
        scaler = self.pipeline.named_steps["scaler"]
        with open(MODEL_CACHE, "w") as f:
            json.dump({
                "coef":          ridge.coef_.tolist(),
                "intercept":     float(ridge.intercept_),
                "scaler_mean":   scaler.mean_.tolist(),
                "scaler_scale":  scaler.scale_.tolist(),
                "feature_names": self.feature_names,
                "cv_r2":         self.cv_r2,
            }, f, indent=2)
        logger.info(f"Player valuation model saved to {MODEL_CACHE}")

    def _load(self):
        with open(MODEL_CACHE) as f:
            data = json.load(f)
        ridge = self.pipeline.named_steps["ridge"]
        scaler = self.pipeline.named_steps["scaler"]
        ridge.coef_      = np.array(data["coef"])
        ridge.intercept_ = data["intercept"]
        scaler.mean_     = np.array(data["scaler_mean"])
        scaler.scale_    = np.array(data["scaler_scale"])
        self.feature_names = data["feature_names"]
        self.cv_r2         = data.get("cv_r2")
        self.is_fitted     = True
