"""
Trade Evaluation Engine — the final layer that answers:
"Who wins this trade and by how much?"

Takes a proposed trade (players + picks per side) and returns:
  - Total value for each team
  - Value differential
  - Win-now vs future value breakdown
  - Verdict string
"""

from dataclasses import dataclass, field
from typing import Optional
from .player_valuation import PlayerValuationResult
from .pick_value import PickValue


@dataclass
class TradeAsset:
    """A single asset in a trade — either a player or a pick."""
    asset_type: str                         # "player" or "pick"

    # Player fields
    player_result: Optional[PlayerValuationResult] = None

    # Pick fields
    pick_result: Optional[PickValue] = None
    pick_year: Optional[int] = None
    pick_round: Optional[int] = None

    # Salary retention
    retained_pct: float = 0.0              # 0.0 to 0.5 (NHL max is 50%)

    @property
    def trade_value(self) -> float:
        if self.asset_type == "player" and self.player_result:
            base = self.player_result.trade_value
            # Retained salary increases the asset's attractiveness to receiver
            retention_bonus = self.retained_pct * 1.5
            return round(base + retention_bonus, 2)
        elif self.asset_type == "pick" and self.pick_result:
            return self.pick_result.value_score
        return 0.0

    @property
    def display_name(self) -> str:
        if self.asset_type == "player" and self.player_result:
            return self.player_result.player_name
        elif self.asset_type == "pick" and self.pick_result:
            rd = {1: "1st", 2: "2nd", 3: "3rd"}.get(self.pick_round, f"{self.pick_round}th")
            return f"{self.pick_year} {rd} Round (#{self.pick_result.overall})"
        return "Unknown asset"

    @property
    def war_now(self) -> float:
        """Current season WAR contribution."""
        if self.asset_type == "player" and self.player_result:
            return self.player_result.war_estimate
        return 0.0

    @property
    def future_war(self) -> float:
        """Projected peak WAR — measures future value."""
        if self.asset_type == "player" and self.player_result:
            return self.player_result.peak_war_proj
        elif self.asset_type == "pick" and self.pick_result:
            return self.pick_result.expected_war
        return 0.0


@dataclass
class TradeSide:
    """One team's side of the trade."""
    team_name: str
    assets: list[TradeAsset] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        return round(sum(a.trade_value for a in self.assets), 2)

    @property
    def win_now_value(self) -> float:
        """Sum of current WAR — measures immediate impact."""
        return round(sum(a.war_now for a in self.assets), 2)

    @property
    def future_value(self) -> float:
        """Sum of projected peak WAR — measures long-term return."""
        return round(sum(a.future_war for a in self.assets), 2)

    @property
    def total_cap_added(self) -> float:
        cap = 0.0
        for a in self.assets:
            if a.asset_type == "player" and a.player_result:
                contract = getattr(a.player_result, "_contract", None)
                if contract:
                    cap += contract.get("cap_hit", 0) * (1 - a.retained_pct)
        return cap


@dataclass
class TradeVerdict:
    side_a: TradeSide
    side_b: TradeSide

    @property
    def differential(self) -> float:
        """Positive = side_a wins. Negative = side_b wins."""
        return round(self.side_a.total_value - self.side_b.total_value, 2)

    @property
    def winner(self) -> str:
        diff = abs(self.differential)
        if diff < 0.5:
            return "Even"
        return self.side_a.team_name if self.differential > 0 else self.side_b.team_name

    @property
    def win_now_winner(self) -> str:
        a = self.side_a.win_now_value
        b = self.side_b.win_now_value
        if abs(a - b) < 0.3:
            return "Even"
        return self.side_a.team_name if a > b else self.side_b.team_name

    @property
    def future_winner(self) -> str:
        a = self.side_a.future_value
        b = self.side_b.future_value
        if abs(a - b) < 0.3:
            return "Even"
        return self.side_a.team_name if a > b else self.side_b.team_name

    @property
    def verdict_label(self) -> str:
        diff = abs(self.differential)
        if diff < 0.5:
            return "Even trade"
        winner = self.winner
        if diff < 1.5:
            return f"{winner} slight edge"
        if diff < 3.0:
            return f"{winner} wins"
        return f"{winner} wins decisively"

    @property
    def bar_pct(self) -> tuple[float, float]:
        """
        Returns (side_a_pct, side_b_pct) for the value bar (0-50 each).
        Both sides share 100%, split proportionally.
        """
        total = self.side_a.total_value + self.side_b.total_value
        if total == 0:
            return 25.0, 25.0
        a_pct = (self.side_a.total_value / total) * 50
        b_pct = (self.side_b.total_value / total) * 50
        return round(a_pct, 1), round(b_pct, 1)

    def to_dict(self) -> dict:
        """Serialize for API response."""
        a_pct, b_pct = self.bar_pct
        return {
            "side_a": {
                "team":         self.side_a.team_name,
                "total_value":  self.side_a.total_value,
                "win_now":      self.side_a.win_now_value,
                "future_value": self.side_a.future_value,
                "assets": [
                    {
                        "name":        a.display_name,
                        "type":        a.asset_type,
                        "trade_value": a.trade_value,
                        "war_now":     a.war_now,
                        "future_war":  a.future_war,
                        "contract_note": (
                            a.player_result.contract_note
                            if a.asset_type == "player" and a.player_result
                            else None
                        ),
                        "age_stage": (
                            a.player_result.age_curve_stage
                            if a.asset_type == "player" and a.player_result
                            else None
                        ),
                    }
                    for a in self.side_a.assets
                ],
            },
            "side_b": {
                "team":         self.side_b.team_name,
                "total_value":  self.side_b.total_value,
                "win_now":      self.side_b.win_now_value,
                "future_value": self.side_b.future_value,
                "assets": [
                    {
                        "name":        a.display_name,
                        "type":        a.asset_type,
                        "trade_value": a.trade_value,
                        "war_now":     a.war_now,
                        "future_war":  a.future_war,
                        "contract_note": (
                            a.player_result.contract_note
                            if a.asset_type == "player" and a.player_result
                            else None
                        ),
                        "age_stage": (
                            a.player_result.age_curve_stage
                            if a.asset_type == "player" and a.player_result
                            else None
                        ),
                    }
                    for a in self.side_b.assets
                ],
            },
            "verdict": {
                "label":          self.verdict_label,
                "winner":         self.winner,
                "differential":   self.differential,
                "win_now_winner": self.win_now_winner,
                "future_winner":  self.future_winner,
                "bar_a_pct":      a_pct,
                "bar_b_pct":      b_pct,
            },
        }
