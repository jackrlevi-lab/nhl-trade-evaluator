"""
NHL Trade Evaluator API
FastAPI application — the layer between your models and the frontend.

Endpoints:
  GET  /players/search?q={name}           — autocomplete player search
  GET  /players/{name}/value              — single player valuation
  GET  /picks/value?overall={n}           — pick value by position
  POST /trades/evaluate                   — full trade evaluation
  GET  /health                            — status + model readiness

Run locally:
  uvicorn nhl_trade_evaluator.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.database import get_session, PlayerSeason, PlayerContract, PlayerWAR, PlayerValuation
from models.pick_value import PickValueModel
from models.player_valuation import PlayerValuationModel
from models.trade_evaluator import (
    TradeAsset, TradeSide, TradeVerdict, PickValue, PlayerValuationResult
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NHL Trade Evaluator",
    description="Player valuation and trade analysis API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Model singletons loaded at startup --
pick_model = PickValueModel()
player_model = PlayerValuationModel()


# -----------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------

class PlayerAssetRequest(BaseModel):
    player_name: str
    season: str = "20232024"
    retained_pct: float = 0.0

class PickAssetRequest(BaseModel):
    overall: int
    year: int
    round: int
    retained_pct: float = 0.0

class TradeSideRequest(BaseModel):
    team_name: str
    players: list[PlayerAssetRequest] = []
    picks: list[PickAssetRequest] = []

class TradeRequest(BaseModel):
    side_a: TradeSideRequest
    side_b: TradeSideRequest


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_player_stats(session, player_name: str, season: str) -> dict | None:
    """Pull EV stats for a player from the DB."""
    row = session.query(PlayerSeason).filter(
        PlayerSeason.player_name.ilike(f"%{player_name}%"),
        PlayerSeason.season == season,
        PlayerSeason.situation == "ev",
    ).first()
    if not row:
        return None
    return {
        "player_name": row.player_name,
        "team":        row.team,
        "position":    row.position,
        "gp":          row.gp,
        "toi":         row.toi,
        "xgf_per60":   row.xgf_per60,
        "xga_per60":   row.xga_per60,
        "rel_cf_pct":  row.rel_cf_pct,
        "ixg":         row.ixg,
        "ozs_pct":     row.ozs_pct,
    }


def get_player_contract(session, player_name: str) -> dict | None:
    """Pull contract data for a player."""
    row = session.query(PlayerContract).filter(
        PlayerContract.player_name.ilike(f"%{player_name}%")
    ).first()
    if not row:
        return None
    return {
        "cap_hit":          row.cap_hit,
        "years_remaining":  row.years_remaining,
        "expiry_status":    row.expiry_status,
        "has_nmc":          row.has_nmc,
        "has_ntc":          row.has_ntc,
    }


def resolve_player_asset(
    session, req: PlayerAssetRequest
) -> TradeAsset | None:
    """Build a TradeAsset from a player request, using DB + model."""
    stats    = get_player_stats(session, req.player_name, req.season)
    contract = get_player_contract(session, req.player_name)

    if not stats:
        logger.warning(f"No stats found for {req.player_name} {req.season}")
        return None

    if not player_model.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Player valuation model not trained yet. Run build_player_valuation_model() first."
        )

    result = player_model.value_player(
        player_name=req.player_name,
        season=req.season,
        nst_row=stats,
        contract=contract,
        age=None,           # TODO: add age to PlayerContract table
        position=stats.get("position"),
    )
    return TradeAsset(
        asset_type="player",
        player_result=result,
        retained_pct=req.retained_pct,
    )


def resolve_pick_asset(req: PickAssetRequest) -> TradeAsset:
    """Build a TradeAsset from a pick request."""
    if not pick_model.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Pick value model not trained yet. Run build_pick_value_model() first."
        )
    pick_val = pick_model.value(req.overall)
    return TradeAsset(
        asset_type="pick",
        pick_result=pick_val,
        pick_year=req.year,
        pick_round=req.round,
        retained_pct=req.retained_pct,
    )


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":              "ok",
        "pick_model_ready":    pick_model.is_fitted,
        "player_model_ready":  player_model.is_fitted,
    }


@app.get("/players/search")
def search_players(q: str = Query(..., min_length=2)):
    """Autocomplete — returns matching player names from DB."""
    session = get_session()
    try:
        rows = session.query(PlayerSeason.player_name, PlayerSeason.team).filter(
            PlayerSeason.player_name.ilike(f"%{q}%"),
            PlayerSeason.situation == "ev",
        ).distinct().limit(10).all()
        return [{"name": r.player_name, "team": r.team} for r in rows]
    finally:
        session.close()


@app.get("/players/{player_name}/value")
def player_value(player_name: str, season: str = "20232024"):
    """Single player valuation."""
    session = get_session()
    try:
        req    = PlayerAssetRequest(player_name=player_name, season=season)
        asset  = resolve_player_asset(session, req)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
        r = asset.player_result
        return {
            "player_name":         r.player_name,
            "team":                r.team,
            "position":            r.position,
            "age":                 r.age,
            "war_estimate":        r.war_estimate,
            "contract_efficiency": r.contract_efficiency,
            "trade_value":         r.trade_value,
            "peak_war_proj":       r.peak_war_proj,
            "age_curve_stage":     r.age_curve_stage,
            "contract_note":       r.contract_note,
        }
    finally:
        session.close()


@app.get("/picks/value")
def pick_value(overall: int = Query(..., ge=1, le=224)):
    """Pick value by overall draft position."""
    if not pick_model.is_fitted:
        raise HTTPException(status_code=503, detail="Pick model not ready")
    pv = pick_model.value(overall)
    return {
        "overall":       pv.overall,
        "expected_war":  pv.expected_war,
        "p_nhl_impact":  pv.p_nhl_impact,
        "value_score":   pv.value_score,
    }


@app.get("/picks/table")
def pick_table():
    """Full pick value table — used to render the pick value curve chart."""
    if not pick_model.is_fitted:
        raise HTTPException(status_code=503, detail="Pick model not ready")
    df = pick_model.value_table()
    return df.to_dict(orient="records")


@app.post("/trades/evaluate")
def evaluate_trade(req: TradeRequest):
    """
    Core endpoint — evaluates a full trade proposal.
    Accepts players and picks for each side, returns full verdict.
    """
    session = get_session()
    try:
        # Resolve side A
        a_assets = []
        for p in req.side_a.players:
            asset = resolve_player_asset(session, p)
            if asset:
                a_assets.append(asset)
        for p in req.side_a.picks:
            a_assets.append(resolve_pick_asset(p))

        # Resolve side B
        b_assets = []
        for p in req.side_b.players:
            asset = resolve_player_asset(session, p)
            if asset:
                b_assets.append(asset)
        for p in req.side_b.picks:
            b_assets.append(resolve_pick_asset(p))

        if not a_assets and not b_assets:
            raise HTTPException(status_code=400, detail="No valid assets found in trade")

        side_a  = TradeSide(team_name=req.side_a.team_name, assets=a_assets)
        side_b  = TradeSide(team_name=req.side_b.team_name, assets=b_assets)
        verdict = TradeVerdict(side_a=side_a, side_b=side_b)

        return verdict.to_dict()

    finally:
        session.close()
