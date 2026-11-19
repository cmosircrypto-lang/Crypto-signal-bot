
from decimal import Decimal

from trading_system import TradingSystem
from okx_contract_specs import ContractSpec


def make_spec(
    inst_id="BTC-USDT-SWAP",
    contract_value="0.01",
    lot_size="0.1",
    min_size="0.1",
):
    return ContractSpec(
        inst_id=inst_id,
        base_ccy="BTC",
        quote_ccy="USDT",
        settle_ccy="USDT",
        contract_type="linear",
        contract_value=Decimal(contract_value),
        contract_value_currency="BTC",
        lot_size=Decimal(lot_size),
        min_size=Decimal(min_size),
        tick_size=Decimal("0.1"),
        state="live",
    )


def make_contract_system():
    return TradingSystem(
        initial_capital=100,
        contract_specs={
            "BTC-USDT-SWAP": make_spec(),
        },
    )


class MockContractSpecs:
    """
    Mimics the get_all() interface of OKXContractSpecs.
    """

    def __init__(self, contracts):
        self.contracts = contracts

    def get_all(self):
        return dict(self.contracts)


# --------------------------------------------------
# Legacy integration tests
# --------------------------------------------------

def test_open_trade():
    system = TradingSystem(initial_capital=100)

    position = system.open_trade(
        symbol="BTC-USDT",
        direction="BUY",
        entry_price=100,
        stop_loss=90,
        take_profit=110,
        risk_percent=1,
        timestamp=1000,
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
        ("LINK-USDT", "SELL"),
    ]

    for symbol, direction in symbols:
        position = system.open_trade(
            symbol,
            direction,
            100,
            90 if direction == "BUY" else 110,
            110 if direction == "BUY" else 90,
            1,
            1000,
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
        2000,
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
        2000,
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


# --------------------------------------------------
# OKX contract integration tests
# --------------------------------------------------

def test_contract_specs_passed_to_execution_engine():
    system = make_contract_system()

    assert system.execution.contract_specs is not None
    assert "BTC-USDT-SWAP" in system.execution.contract_specs

    print("PASS: contract specifications connected to execution")


def test_contract_trade_opened_through_trading_system():
    system = make_contract_system()

    position = system.open_trade(
        symbol="BTC-USDT-SWAP",
        direction="BUY",
        entry_price=50000,
        stop_loss=49000,
        take_profit=52000,
        risk_percent=1,
        timestamp=1000,
    )

    assert position is not None
    assert abs(position.contract_count - 0.1) < 1e-9
    assert abs(position.quantity - 0.001) < 1e-9
    assert abs(position.risk_amount - 1.0) < 1e-9

    assert "BTC-USDT-SWAP" in system.execution.open_positions
    assert system.portfolio.positions["BTC-USDT-SWAP"] == "BUY"

    print("PASS: contract-based trade opened through TradingSystem")


def test_contract_trade_exit_releases_capacity():
    system = make_contract_system()

    system.open_trade(
        "BTC-USDT-SWAP",
        "BUY",
        50000,
        49000,
        52000,
        1,
        1000,
    )

    result = system.check_exit(
        "BTC-USDT-SWAP",
        {"high": 52100, "low": 50100},
        2000,
    )

    assert result is not None
    assert result.exit_reason == "TP"
    assert abs(result.pnl - 2.0) < 1e-9
    assert abs(system.execution.equity - 102.0) < 1e-9

    assert "BTC-USDT-SWAP" not in system.execution.open_positions
    assert "BTC-USDT-SWAP" not in system.portfolio.positions
    assert system.can_search_symbol("BTC-USDT-SWAP")

    print("PASS: contract trade exit releases portfolio capacity")


def test_missing_contract_spec_rolls_back_portfolio():
    system = make_contract_system()

    try:
        system.open_trade(
            "ETH-USDT-SWAP",
            "BUY",
            3000,
            2900,
            3200,
            1,
            1000,
        )
        raise AssertionError(
            "Trade without contract specification was accepted"
        )
    except KeyError:
        pass

    assert "ETH-USDT-SWAP" not in system.execution.open_positions
    assert "ETH-USDT-SWAP" not in system.portfolio.positions
    assert system.can_search_symbol("ETH-USDT-SWAP")

    print("PASS: failed contract trade rolls back portfolio reservation")


def test_contract_specs_object_interface():
    specs = MockContractSpecs({
        "BTC-USDT-SWAP": make_spec(),
    })

    system = TradingSystem(
        initial_capital=100,
        contract_specs=specs,
    )

    position = system.open_trade(
        "BTC-USDT-SWAP",
        "BUY",
        50000,
        49000,
        52000,
        1,
        1000,
    )

    assert position is not None
    assert abs(position.contract_count - 0.1) < 1e-9

    print("PASS: contract specs object interface supported")


# --------------------------------------------------
# Test runner
# --------------------------------------------------

def main():
    test_open_trade()
    test_duplicate_symbol_blocked()
    test_portfolio_limits()
    test_take_profit_releases_capacity()
    test_stop_loss_releases_capacity()
    test_manual_close_releases_capacity()
    test_status_consistency()

    test_contract_specs_passed_to_execution_engine()
    test_contract_trade_opened_through_trading_system()
    test_contract_trade_exit_releases_capacity()
    test_missing_contract_spec_rolls_back_portfolio()
    test_contract_specs_object_interface()

    print("ALL TRADING SYSTEM TESTS PASSED")


if __name__ == "__main__":
    main()
