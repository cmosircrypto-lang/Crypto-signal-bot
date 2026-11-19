from execution_engine import ExecutionEngine
from portfolio_engine import PortfolioEngine


class TradingSystem:
    """
    Coordinates ExecutionEngine and PortfolioEngine.
    """

    def __init__(
        self,
        initial_capital=100.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        max_total=6,
        max_buy=3,
        max_sell=3
    ):
        self.execution = ExecutionEngine(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate
        )

        self.portfolio = PortfolioEngine(
            max_total=max_total,
            max_buy=max_buy,
            max_sell=max_sell
        )

    @staticmethod
    def normalize_symbol(symbol):
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Invalid symbol")

        return symbol.strip().upper()

    def can_search_symbol(self, symbol):
        """
        A symbol with an open position must not be
        searched for new trading signals.
        """

        symbol = self.normalize_symbol(symbol)

        return (
            symbol not in self.execution.open_positions
            and symbol not in self.portfolio.positions
        )

    def can_open_trade(self, symbol, direction):
        """
        Checks portfolio capacity before opening a trade.
        """

        symbol = self.normalize_symbol(symbol)

        if not self.can_search_symbol(symbol):
            return False

        return self.portfolio.can_open(symbol, direction)

    def open_trade(
        self,
        symbol,
        direction,
        entry_price,
        stop_loss,
        take_profit,
        risk_percent,
        timestamp
    ):
        """
        Registers portfolio capacity and opens the trade.

        If execution fails, the portfolio slot is released.
        """

        symbol = self.normalize_symbol(symbol)
        direction = direction.strip().upper()

        if not self.can_open_trade(symbol, direction):
            return None

        # Reserve portfolio capacity first.
        registered = self.portfolio.open_position(
            symbol,
            direction
        )

        if not registered:
            return None

        try:
            position = self.execution.open_trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_percent=risk_percent,
                timestamp=timestamp
            )

            return position

        except Exception:
            # Roll back the portfolio reservation.
            self.portfolio.close_position(symbol)
            raise

    def check_exit(self, symbol, candle, timestamp):
        """
        Checks SL/TP through ExecutionEngine.

        When a trade closes, portfolio capacity is released.
        """

        symbol = self.normalize_symbol(symbol)

        result = self.execution.check_exit(
            symbol,
            candle,
            timestamp
        )

        if result is not None:
            self.portfolio.close_position(symbol)

        return result

    def close_trade(
        self,
        symbol,
        exit_price,
        timestamp,
        exit_reason
    ):
        """
        Manually closes a trade and releases its capacity.
        """

        symbol = self.normalize_symbol(symbol)

        result = self.execution.close_trade(
            symbol,
            exit_price,
            timestamp,
            exit_reason
        )

        self.portfolio.close_position(symbol)

        return result

    def get_status(self):
        """
        Returns combined execution and portfolio status.
        """

        return {
            "portfolio": self.portfolio.get_status(),
            "execution": self.execution.get_summary(),
            "open_positions": list(
                self.execution.open_positions.keys()
            ),
            "closed_trades": len(
                self.execution.closed_trades
            )
        }