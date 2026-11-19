class PortfolioEngine:
    """
    Controls portfolio capacity and prevents duplicate positions.
    """

    def __init__(
        self,
        max_total=6,
        max_buy=3,
        max_sell=3
    ):
        if max_total < 1 or max_buy < 0 or max_sell < 0:
            raise ValueError("Invalid portfolio limits")

        if max_buy + max_sell < max_total:
            raise ValueError(
                "Direction limits cannot exceed total capacity"
            )

        self.max_total = max_total
        self.max_buy = max_buy
        self.max_sell = max_sell

        # symbol -> direction
        self.positions = {}

    @staticmethod
    def _normalize_symbol(symbol):
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Invalid symbol")

        return symbol.strip().upper()

    @staticmethod
    def _normalize_direction(direction):
        if not isinstance(direction, str):
            raise ValueError("Invalid direction")

        direction = direction.strip().upper()

        if direction not in ("BUY", "SELL"):
            raise ValueError("Direction must be BUY or SELL")

        return direction

    def can_open(self, symbol, direction):
        """
        Checks whether a new position is allowed.

        This check should happen BEFORE searching for a signal
        on the requested symbol.
        """

        symbol = self._normalize_symbol(symbol)
        direction = self._normalize_direction(direction)

        # One open position per symbol
        if symbol in self.positions:
            return False

        # Total portfolio capacity
        if len(self.positions) >= self.max_total:
            return False

        buy_count = sum(
            1 for side in self.positions.values()
            if side == "BUY"
        )

        sell_count = sum(
            1 for side in self.positions.values()
            if side == "SELL"
        )

        # Direction limits
        if direction == "BUY" and buy_count >= self.max_buy:
            return False

        if direction == "SELL" and sell_count >= self.max_sell:
            return False

        return True

    def open_position(self, symbol, direction):
        """
        Registers a position after the execution engine
        successfully opens the actual simulated trade.

        Returns True if registered, otherwise False.
        """

        symbol = self._normalize_symbol(symbol)
        direction = self._normalize_direction(direction)

        if not self.can_open(symbol, direction):
            return False

        self.positions[symbol] = direction
        return True

    def close_position(self, symbol):
        """
        Releases the portfolio slot after a trade is closed.
        """

        symbol = self._normalize_symbol(symbol)

        if symbol not in self.positions:
            return False

        del self.positions[symbol]
        return True

    def get_open_symbols(self):
        return list(self.positions.keys())

    def get_status(self):
        buy_count = sum(
            1 for side in self.positions.values()
            if side == "BUY"
        )

        sell_count = sum(
            1 for side in self.positions.values()
            if side == "SELL"
        )

        return {
            "total_open": len(self.positions),
            "buy_open": buy_count,
            "sell_open": sell_count,
            "remaining_capacity": (
                self.max_total - len(self.positions)
            ),
            "open_symbols": self.get_open_symbols()
        }