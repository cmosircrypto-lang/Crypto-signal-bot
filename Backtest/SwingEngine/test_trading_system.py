from trading_system import TradingSystem


def test_open_trade():
    system = TradingSystem(initial_capital=100)

    position = system.open_trade(
        symbol="BTC-USDT",
        direction="BUY",
        entry_price=100,
        stop_loss=90,
        take_profit=110,
        risk_percent=1,
        timestamp=1000
    )

    assert position is not None
    assert "BTC-USDT" in system.execution.open_positions
    assert system.portfolio.positions["BTC-USDT"] == "BUY"

    print("PASS: trade opened in both engines")


def test_duplicate_symbol_blocked():
    system = TradingSystem()

    system.open_trade(
        "BTC-USDT", "BUY", 100, 90, 110, 1, 1000
    )

    assert not system.can_search_symbol("BTC-USDT")

    duplicate = system.open_trade(
        "BTC-USDT", "SELL", 100, 110, 90, 1, 2000
    )

    assert duplicate is None

    print("PASS: duplicate symbol blocked")


def test_portfolio_limits():
    system = TradingSystem()

    symbols = [
        ("BTC-USDT", "BUY"),
        ("ETH-USDT", "BUY"),
        ("SOL-USDT", "BUY"),
        ("XRP-USDT", "SELL"),
        ("ADA-USDT", "SELL"),
        ("LINK-USDT", "SELL")
    ]

    for symbol, direction in symbols:
        position = system.open_trade(
            symbol, direction, 100, 90 if direction == "BUY" else 110,
            110 if direction == "BUY" else 90, 1, 1000
        )
        assert position is not None

    assert len(system.execution.open_positions) == 6
    assert system.portfolio.get_status()["buy_open"] == 3
    assert system.portfolio.get_status()["sell_open"] == 3

    rejected = system.open_trade(
        "AVAX-USDT", "BUY", 100, 90, 110, 1, 2000
    )

    assert rejected is None

    print("PASS: integrated six-position and direction limits")


def test_take_profit_releases_capacity():
    system = TradingSystem()

    system.open_trade(
        "BTC-USDT", "BUY", 100, 90, 110, 1, 1000
    )

    result = system.check_exit(
        "BTC-USDT",
        {"high": 111, "low": 99},
        2000
    )

    assert result is not None
    assert result.exit_reason == "TP"

    assert "BTC-USDT" not in system.execution.open_positions
    assert "BTC-USDT" not in system.portfolio.positions
    assert system.can_search_symbol("BTC-USDT")

    print("PASS: take-profit closes trade and releases capacity")


def test_stop_loss_releases_capacity():
    system = TradingSystem()

    system.open_trade(
        "BTC-USDT", "BUY", 100, 90, 110, 1, 1000
    )

    result = system.check_exit(
        "BTC-USDT",
        {"high": 101, "low": 89},
        2000
    )

    assert result is not None
    assert result.exit_reason == "SL"

    assert "BTC-USDT" not in system.execution.open_positions
    assert "BTC-USDT" not in system.portfolio.positions

    print("PASS: stop-loss closes trade and releases capacity")


def test_manual_close_releases_capacity():
    system = TradingSystem()

    system.open_trade(
        "ETH-USDT", "BUY", 100, 90, 110, 1, 1000
    )

    result = system.close_trade(
        "ETH-USDT", 105, 2000, "MANUAL"
    )

    assert result is not None
    assert "ETH-USDT" not in system.execution.open_positions
    assert "ETH-USDT" not in system.portfolio.positions
    assert system.can_search_symbol("ETH-USDT")

    print("PASS: manual close releases capacity")


def test_status_consistency():
    system = TradingSystem()

    system.open_trade(
        "BTC-USDT", "BUY", 100, 90, 110, 1, 1000
    )

    system.open_trade(
        "ETH-USDT", "SELL", 100, 110, 90, 1, 1000
    )

    status = system.get_status()

    assert status["portfolio"]["total_open"] == 2
    assert status["portfolio"]["buy_open"] == 1
    assert status["portfolio"]["sell_open"] == 1
    assert len(status["open_positions"]) == 2
    assert status["execution"]["open_trades"] == 2

    print("PASS: execution and portfolio status synchronized")


if __name__ == "__main__":
    test_open_trade()
    test_duplicate_symbol_blocked()
    test_portfolio_limits()
    test_take_profit_releases_capacity()
    test_stop_loss_releases_capacity()
    test_manual_close_releases_capacity()
    test_status_consistency()

    print("ALL TRADING SYSTEM TESTS PASSED")