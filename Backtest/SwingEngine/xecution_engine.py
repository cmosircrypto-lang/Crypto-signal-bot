
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import math


INITIAL_CAPITAL = 100.0


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_percent: float
    risk_amount: float
    opened_at: int
    status: str = "OPEN"


@dataclass
class TradeResult:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    risk_percent: float
    pnl: float
    result: str
    exit_reason: str
    opened_at: int
    closed_at: int


class ExecutionEngine:
    """
    Simulation-only execution engine.

    Rules:
    - Initial capital: $100
    - Risk based on current equity.
    - No leverage-based risk inflation.
    - Fees and slippage are zero for the initial tests.
    - If SL and TP are both touched in one candle,
      SL is assumed to execute first.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        fee_rate: float = 0.0,
        slippage_rate: float = 0.0,
    ):
        if not math.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        if not 0 <= fee_rate < 1:
            raise ValueError("Invalid fee rate")

        if not 0 <= slippage_rate < 1:
            raise ValueError("Invalid slippage rate")

        self.initial_capital = float(initial_capital)
        self.equity = float(initial_capital)
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)

        self.open_positions: dict[str, Position] = {}
        self.closed_trades: list[TradeResult] = []

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_percent: float,
    ) -> tuple[float, float]:
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError("Invalid entry price")

        if not math.isfinite(stop_loss) or stop_loss <= 0:
            raise ValueError("Invalid stop loss")

        if not 0 < risk_percent <= 100:
            raise ValueError("Invalid risk percentage")

        distance = abs(entry_price - stop_loss)

        if distance <= 0:
            raise ValueError("Entry and stop loss cannot be equal")

        risk_amount = self.equity * risk_percent / 100
        quantity = risk_amount / distance

        return quantity, risk_amount

    def open_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_percent: float,
        timestamp: int,
    ) -> Position:
        if symbol in self.open_positions:
            raise ValueError("Symbol already has an open position")

        if direction not in ("BUY", "SELL"):
            raise ValueError("Direction must be BUY or SELL")

        prices = (entry_price, stop_loss, take_profit)

        if not all(math.isfinite(p) and p > 0 for p in prices):
            raise ValueError("Prices must be finite and positive")

        if direction == "BUY":
            if not stop_loss < entry_price < take_profit:
                raise ValueError("Invalid BUY price structure")
        else:
            if not take_profit < entry_price < stop_loss:
                raise ValueError("Invalid SELL price structure")

        quantity, risk_amount = self.calculate_position_size(
            entry_price,
            stop_loss,
            risk_percent,
        )

        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            quantity=quantity,
            risk_percent=float(risk_percent),
            risk_amount=risk_amount,
            opened_at=int(timestamp),
        )

        self.open_positions[symbol] = position
        return position

    def check_exit(
        self,
        symbol: str,
        candle: dict,
        timestamp: int,
    ) -> Optional[TradeResult]:
        if symbol not in self.open_positions:
            return None

        position = self.open_positions[symbol]

        high = float(candle["high"])
        low = float(candle["low"])

        if not math.isfinite(high) or not math.isfinite(low):
            raise ValueError("Invalid candle prices")

        if high < low:
            raise ValueError("Candle high cannot be below low")

        if position.direction == "BUY":
            sl_hit = low <= position.stop_loss
            tp_hit = high >= position.take_profit
        else:
            sl_hit = high >= position.stop_loss
            tp_hit = low <= position.take_profit

        if sl_hit:
            return self.close_trade(
                symbol,
                position.stop_loss,
                timestamp,
                "SL",
            )

        if tp_hit:
            return self.close_trade(
                symbol,
                position.take_profit,
                timestamp,
                "TP",
            )

        return None

    def close_trade(
        self,
        symbol: str,
        exit_price: float,
        timestamp: int,
        exit_reason: str,
    ) -> TradeResult:
        if symbol not in self.open_positions:
            raise ValueError("No open position for symbol")

        position = self.open_positions[symbol]

        if not math.isfinite(exit_price) or exit_price <= 0:
            raise ValueError("Invalid exit price")

        if exit_reason not in ("SL", "TP", "MANUAL"):
            raise ValueError("Invalid exit reason")

        if position.direction == "BUY":
            raw_pnl = (
                exit_price - position.entry_price
            ) * position.quantity
        else:
            raw_pnl = (
                position.entry_price - exit_price
            ) * position.quantity

        entry_fee = (
            position.entry_price
            * position.quantity
            * self.fee_rate
        )

        exit_fee = (
            exit_price
            * position.quantity
            * self.fee_rate
        )

        fees = entry_fee + exit_fee

        slippage_cost = (
            position.entry_price + exit_price
        ) * position.quantity * self.slippage_rate

        pnl = raw_pnl - fees - slippage_cost

        self.equity += pnl

        result = TradeResult(
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=float(exit_price),
            quantity=position.quantity,
            risk_percent=position.risk_percent,
            pnl=pnl,
            result="WIN" if pnl > 0 else (
                "LOSS" if pnl < 0 else "BREAKEVEN"
            ),
            exit_reason=exit_reason,
            opened_at=position.opened_at,
            closed_at=int(timestamp),
        )

        position.status = "CLOSED"

        del self.open_positions[symbol]
        self.closed_trades.append(result)

        return result

    def get_summary(self) -> dict:
        wins = sum(
            trade.pnl > 0 for trade in self.closed_trades
        )

        losses = sum(
            trade.pnl < 0 for trade in self.closed_trades
        )

        total_pnl = sum(
            trade.pnl for trade in self.closed_trades
        )

        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.equity,
            "total_pnl": total_pnl,
            "closed_trades": len(self.closed_trades),
            "open_trades": len(self.open_positions),
            "wins": wins,
            "losses": losses,
            "win_rate": (
                wins / len(self.closed_trades) * 100
                if self.closed_trades else 0.0
            ),
        }

    def export_trades(self) -> list[dict]:
        return [
            asdict(trade)
            for trade in self.closed_trades
        ]
