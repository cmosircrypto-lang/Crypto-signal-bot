from portfolio_engine import PortfolioEngine


def test_default_limits():
    portfolio = PortfolioEngine()

    assert portfolio.max_total == 6
    assert portfolio.max_buy == 3
    assert portfolio.max_sell == 3

    print("PASS: default portfolio limits")


def test_maximum_three_buys():
    portfolio = PortfolioEngine()

    assert portfolio.open_position("BTC", "BUY")
    assert portfolio.open_position("ETH", "BUY")
    assert portfolio.open_position("SOL", "BUY")

    assert not portfolio.can_open("XRP", "BUY")

    print("PASS: maximum three BUY positions")


def test_maximum_three_sells():
    portfolio = PortfolioEngine()

    assert portfolio.open_position("BTC", "SELL")
    assert portfolio.open_position("ETH", "SELL")
    assert portfolio.open_position("SOL", "SELL")

    assert not portfolio.can_open("XRP", "SELL")

    print("PASS: maximum three SELL positions")


def test_maximum_six_positions():
    portfolio = PortfolioEngine()

    symbols = ["BTC", "ETH", "SOL", "XRP", "ADA", "LINK"]
    directions = ["BUY", "BUY", "BUY", "SELL", "SELL", "SELL"]

    for symbol, direction in zip(symbols, directions):
        assert portfolio.open_position(symbol, direction)

    assert len(portfolio.positions) == 6
    assert not portfolio.can_open("AVAX", "BUY")
    assert not portfolio.can_open("AVAX", "SELL")

    print("PASS: maximum six open positions")


def test_one_position_per_symbol():
    portfolio = PortfolioEngine()

    assert portfolio.open_position("BTC", "BUY")
    assert not portfolio.can_open("BTC", "SELL")
    assert not portfolio.open_position("BTC", "SELL")

    print("PASS: one open position per symbol")


def test_closed_symbol_can_reopen():
    portfolio = PortfolioEngine()

    assert portfolio.open_position("BTC", "BUY")
    assert portfolio.close_position("BTC")

    assert portfolio.can_open("BTC", "SELL")
    assert portfolio.open_position("BTC", "SELL")

    print("PASS: closed symbol can open a new position")


def test_capacity_released_after_close():
    portfolio = PortfolioEngine()

    symbols = ["BTC", "ETH", "SOL", "XRP", "ADA", "LINK"]
    directions = ["BUY", "BUY", "BUY", "SELL", "SELL", "SELL"]

    for symbol, direction in zip(symbols, directions):
        assert portfolio.open_position(symbol, direction)

    assert portfolio.close_position("BTC")
    assert portfolio.can_open("AVAX", "BUY")

    print("PASS: capacity released after closing a position")


def test_status_report():
    portfolio = PortfolioEngine()

    portfolio.open_position("BTC", "BUY")
    portfolio.open_position("ETH", "BUY")
    portfolio.open_position("SOL", "SELL")

    status = portfolio.get_status()

    assert status["total_open"] == 3
    assert status["buy_open"] == 2
    assert status["sell_open"] == 1
    assert status["remaining_capacity"] == 3

    print("PASS: portfolio status reporting")


def test_invalid_inputs():
    portfolio = PortfolioEngine()

    try:
        portfolio.open_position("BTC", "HOLD")
        assert False
    except ValueError:
        pass

    try:
        portfolio.open_position("", "BUY")
        assert False
    except ValueError:
        pass

    print("PASS: invalid inputs rejected")


if __name__ == "__main__":
    test_default_limits()
    test_maximum_three_buys()
    test_maximum_three_sells()
    test_maximum_six_positions()
    test_one_position_per_symbol()
    test_closed_symbol_can_reopen()
    test_capacity_released_after_close()
    test_status_report()
    test_invalid_inputs()

    print("ALL PORTFOLIO ENGINE TESTS PASSED")