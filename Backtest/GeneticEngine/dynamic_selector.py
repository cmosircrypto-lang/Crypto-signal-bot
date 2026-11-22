"""Cycle-by-cycle candidate scoring and per-symbol signal selection.

This module is deliberately independent of the production SwingEngine. It ranks
only candidates that already have a valid current signal; historical quality
alone can never create a signal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable


@dataclass(frozen=True)
class ScoringPolicy:
    min_trades: int = 30
    max_drawdown_pct: float = 20.0
    min_profit_factor: float = 1.0
    min_positive_window_ratio: float = 0.5
    min_historical_score: float = 0.45
    min_current_score: float = 0.50
    historical_weight: float = 0.70
    current_weight: float = 0.30

    def validate(self) -> None:
        if self.min_trades < 1:
            raise ValueError("min_trades must be >= 1")
        if self.max_drawdown_pct <= 0:
            raise ValueError("max_drawdown_pct must be > 0")
        if self.min_profit_factor < 0:
            raise ValueError("min_profit_factor must be >= 0")
        for name in ("min_positive_window_ratio", "min_historical_score",
                     "min_current_score", "historical_weight", "current_weight"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isclose(self.historical_weight + self.current_weight, 1.0):
            raise ValueError("historical_weight + current_weight must equal 1")


@dataclass(frozen=True)
class HistoricalMetrics:
    """Metrics must come from train/rolling-validation data, never final OOS."""
    average_r: float
    max_drawdown_pct: float
    profit_factor: float
    positive_window_ratio: float
    trade_count: int
    complexity_penalty: float = 0.0

    def validate(self) -> None:
        values = (self.average_r, self.max_drawdown_pct, self.profit_factor,
                  self.positive_window_ratio, self.complexity_penalty)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("historical metrics must be finite")
        if self.max_drawdown_pct < 0 or self.profit_factor < 0:
            raise ValueError("drawdown and profit factor cannot be negative")
        if self.trade_count < 0:
            raise ValueError("trade_count cannot be negative")
        if not 0 <= self.positive_window_ratio <= 1:
            raise ValueError("positive_window_ratio must be in [0, 1]")
        if not 0 <= self.complexity_penalty <= 1:
            raise ValueError("complexity_penalty must be in [0, 1]")


@dataclass(frozen=True)
class SignalCandidate:
    symbol: str
    candidate_id: str
    method: str
    direction: str
    signal_timestamp: str
    historical: HistoricalMetrics
    current_setup_quality: float
    has_current_signal: bool = True
    candle_closed: bool = True
    rejection_reason: str | None = None

    def validate(self) -> None:
        if not self.symbol.strip() or not self.candidate_id.strip() or not self.method.strip():
            raise ValueError("symbol, candidate_id and method are required")
        if self.direction.upper() not in ("LONG", "SHORT"):
            raise ValueError("direction must be LONG or SHORT")
        if not 0 <= self.current_setup_quality <= 1:
            raise ValueError("current_setup_quality must be in [0, 1]")
        self.historical.validate()


@dataclass(frozen=True)
class ScoredCandidate:
    symbol: str
    candidate_id: str
    method: str
    direction: str
    signal_timestamp: str
    historical_score: float
    current_score: float
    final_score: float
    eligible: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionDecision:
    symbol: str
    winner: ScoredCandidate | None
    ranked_candidates: tuple[ScoredCandidate, ...] = ()

    @property
    def status(self) -> str:
        return "SELECTED" if self.winner is not None else "NO_SIGNAL"


def historical_quality_score(metrics: HistoricalMetrics) -> float:
    """Return a normalized 0..1 score from historical validation metrics."""
    metrics.validate()
    # Saturating transform prevents one extreme average-R result dominating.
    return_score = (math.tanh(metrics.average_r / 2.0) + 1.0) / 2.0
    drawdown_score = max(0.0, 1.0 - metrics.max_drawdown_pct / 30.0)
    pf_score = min(metrics.profit_factor / 2.0, 1.0)
    consistency_score = metrics.positive_window_ratio
    trade_score = min(metrics.trade_count / 100.0, 1.0)
    complexity_score = 1.0 - metrics.complexity_penalty
    score = (
        0.25 * return_score
        + 0.20 * drawdown_score
        + 0.15 * pf_score
        + 0.15 * consistency_score
        + 0.10 * trade_score
        + 0.10 * return_score  # validation return component; final OOS is excluded
        + 0.05 * complexity_score
    )
    return min(1.0, max(0.0, score))


def score_candidate(candidate: SignalCandidate,
                    policy: ScoringPolicy = ScoringPolicy()) -> ScoredCandidate:
    policy.validate()
    candidate.validate()
    metrics = candidate.historical
    historical_score = historical_quality_score(metrics)
    current_score = candidate.current_setup_quality
    reasons: list[str] = []
    if not candidate.has_current_signal:
        reasons.append("NO_CURRENT_SIGNAL")
    if not candidate.candle_closed:
        reasons.append("CANDLE_NOT_CLOSED")
    if candidate.rejection_reason:
        reasons.append(candidate.rejection_reason)
    if metrics.trade_count < policy.min_trades:
        reasons.append("INSUFFICIENT_TRADES")
    if metrics.max_drawdown_pct > policy.max_drawdown_pct:
        reasons.append("DRAWDOWN_LIMIT")
    if metrics.profit_factor < policy.min_profit_factor:
        reasons.append("PROFIT_FACTOR_GATE")
    if metrics.positive_window_ratio < policy.min_positive_window_ratio:
        reasons.append("INCONSISTENT_VALIDATION")
    if historical_score < policy.min_historical_score:
        reasons.append("HISTORICAL_SCORE_GATE")
    if current_score < policy.min_current_score:
        reasons.append("CURRENT_SETUP_SCORE_GATE")
    final_score = (
        policy.historical_weight * historical_score
        + policy.current_weight * current_score
    )
    return ScoredCandidate(
        symbol=candidate.symbol.strip().upper(),
        candidate_id=candidate.candidate_id,
        method=candidate.method,
        direction=candidate.direction.upper(),
        signal_timestamp=candidate.signal_timestamp,
        historical_score=historical_score,
        current_score=current_score,
        final_score=final_score,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def select_cycle_signals(
    candidates: Iterable[SignalCandidate],
    policy: ScoringPolicy = ScoringPolicy(),
) -> dict[str, SelectionDecision]:
    """Select at most one current signal per symbol for this cycle.

    Ties are deterministic: candidate_id ascending after final score descending.
    Symbols with no eligible current signal receive an explicit NO_SIGNAL result.
    """
    grouped: dict[str, list[ScoredCandidate]] = {}
    for candidate in candidates:
        scored = score_candidate(candidate, policy)
        grouped.setdefault(scored.symbol, []).append(scored)

    decisions: dict[str, SelectionDecision] = {}
    for symbol in sorted(grouped):
        ranked = sorted(
            grouped[symbol],
            key=lambda item: (-item.final_score, item.candidate_id),
        )
        winner = next((item for item in ranked if item.eligible), None)
        decisions[symbol] = SelectionDecision(
            symbol=symbol, winner=winner, ranked_candidates=tuple(ranked)
        )
    return decisions
