"""Initial portfolio rules for Swing Engine.

This module intentionally does not generate trading signals yet. It provides
and tests the agreed open-position constraints before strategy logic is added.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TradeSlot:
    symbol: str
    side: str
    trade_id: str


class PortfolioLimitError(ValueError):
    pass


class PortfolioManager:
    """Enforce six total positions: at most three BUY and three SELL.

    A symbol with an open position is locked. The scanner should call
    should_scan(symbol) before doing any signal/OB work for that symbol.
    """

    def __init__(self, max_total: int = 6, max_per_side: int = 3):
        if max_total < 1 or max_per_side < 1:
            raise ValueError("Portfolio limits must be positive")
        if max_total > 2 * max_per_side:
            raise ValueError("max_total cannot exceed both side limits combined")
        self.max_total = max_total
        self.max_per_side = max_per_side
        self._open_by_symbol: Dict[str, TradeSlot] = {}

    @staticmethod
    def _normalize_side(side: str) -> str:
        value = side.strip().upper()
        if value in {"BUY", "LONG"}:
            return "BUY"
        if value in {"SELL", "SHORT"}:
            return "SELL"
        raise ValueError(f"Unsupported trade side: {side}")

    @property
    def open_trades(self) -> tuple[TradeSlot, ...]:
        return tuple(self._open_by_symbol.values())

    @property
    def open_count(self) -> int:
        return len(self._open_by_symbol)

    def count_side(self, side: str) -> int:
        normalized = self._normalize_side(side)
        return sum(t.side == normalized for t in self._open_by_symbol.values())

    def has_open_trade(self, symbol: str) -> bool:
        return symbol in self._open_by_symbol

    def should_scan(self, symbol: str) -> bool:
        """False means do not search for any new signal on this symbol."""
        return not self.has_open_trade(symbol)

    def can_open(self, symbol: str, side: str) -> tuple[bool, str]:
        normalized = self._normalize_side(side)

        if self.has_open_trade(symbol):
            return False, "SYMBOL_ALREADY_OPEN"
        if self.open_count >= self.max_total:
            return False, "PORTFOLIO_TOTAL_LIMIT"
        if self.count_side(normalized) >= self.max_per_side:
            return False, f"{normalized}_LIMIT"
        return True, "OK"

    def open_trade(self, symbol: str, side: str, trade_id: str) -> TradeSlot:
        allowed, reason = self.can_open(symbol, side)
        if not allowed:
            raise PortfolioLimitError(f"Cannot open {symbol} {side}: {reason}")

        slot = TradeSlot(
            symbol=symbol,
            side=self._normalize_side(side),
            trade_id=trade_id,
        )
        self._open_by_symbol[symbol] = slot
        return slot

    def close_trade(self, symbol: str, trade_id: str | None = None) -> TradeSlot:
        if symbol not in self._open_by_symbol:
            raise KeyError(f"No open trade for {symbol}")

        current = self._open_by_symbol[symbol]
        if trade_id is not None and current.trade_id != trade_id:
            raise ValueError(
                f"Trade ID mismatch for {symbol}: "
                f"expected {current.trade_id}, got {trade_id}"
            )

        del self._open_by_symbol[symbol]
        return current

    def snapshot(self) -> dict:
        return {
            "open_total": self.open_count,
            "open_buy": self.count_side("BUY"),
            "open_sell": self.count_side("SELL"),
            "locked_symbols": sorted(self._open_by_symbol),
            "max_total": self.max_total,
            "max_buy": self.max_per_side,
            "max_sell": self.max_per_side,
        }
