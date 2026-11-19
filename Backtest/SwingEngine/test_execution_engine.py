
from execution_engine import ExecutionEngine


def test_position_size():
    engine = ExecutionEngine(initial_capital=100)

    quantity, risk = engine.calculate_position_size(
        entry_price=100,
        stop_loss=98,
        risk_percent=1,
    )

    assert risk == 1.0
    assert quantity == 0.5

    print("PASS: position size and risk calculation")


def test_buy_take_profit():
    engine = ExecutionEngine(initial_capital=100)

    engine.open_trade(
        symbol="BTC-USDT",
        direction="BUY",
        entry_price=100,
        stop_loss=98,
        take_profit=104,
        risk_percent=1,
        timestamp=1000,
    )

    result = engine.check_exit(
        "BTC-USDT",
        {"high": 105, "low": 99},
        2000,
    )

    assert result is not None
    assert result.exit_reason == "TP"
    assert abs(result.pnl - 2.0) < 1e-9
    assert engine.equity == 102.0

    print("PASS: BUY take-profit execution")


def test_sell_stop_loss():
    engine = ExecutionEngine(initial_capital=100)

    engine.open_trade(
        symbol="ETH-USDT",
        direction="SELL",
        entry_price=100,
        stop_loss=102,
        take_profit=96,
        risk_percent=1,
        timestamp=1000,
    )

    result = engine.check_exit(
        "ETH-USDT",
        {"high": 103, "low": 99},
        2000,
    )

    assert result is not None
    assert result.exit_reason == "SL"
    assert abs(result.pnl + 1.0) < 1e-9
    assert engine.equity == 99.0

    print("PASS: SELL stop-loss execution")


def test_sl_tp_same_candle():
    engine = ExecutionEngine(initial_capital=100)

    engine.open_trade(
        symbol="SOL-USDT",
        direction="BUY",
        entry_price=100,
        stop_loss=98,
        take_profit=104,
        risk_percent=1,
        timestamp=1000,
    )

    result = engine.check_exit(
        "SOL-USDT",
        {"high": 105, "low": 97},
        2000,
    )

    assert result is not None
    assert result.exit_reason == "SL"
    assert abs(result.pnl + 1.0) < 1e-9

    print("PASS: simultaneous SL and TP resolves to SL")


def test_one_position_per_symbol():
    engine = ExecutionEngine(initial_capital=100)

    engine.open_trade(
        symbol="BTC-USDT",
        direction="BUY",
        entry_price=100,
        stop_loss=98,
        take_profit=104,
        risk_percent=1,
        timestamp=1000,
    )

    try:
        engine.open_trade(
            symbol="BTC-USDT",
            direction="SELL",
            entry_price=100,
            stop_loss=102,
            take_profit=96,
            risk_percent=1,
            timestamp=2000,
        )
        raise AssertionError("Duplicate symbol position was accepted")
    except ValueError:
        pass

    print("PASS: one open position per symbol")


def test_dynamic_equity_risk():
    engine = ExecutionEngine(initial_capital=100)

    engine.open_trade(
        symbol="BTC-USDT",
        direction="BUY",
        entry_price=100,
        stop_loss=98,
        take_profit=104,
        risk_percent=1,
        timestamp=1000,
    )

    engine.close_trade(
        "BTC-USDT",
        104,
        2000,
        "TP",
    )

    assert abs(engine.equity - 102.0) < 1e-9

    quantity, risk = engine.calculate_position_size(
        entry_price=100,
        stop_loss=98,
        risk_percent=1,
    )

    assert abs(risk - 1.02) < 1e-9
    assert abs(quantity - 0.51) < 1e-9

    print("PASS: risk recalculated from current equity")


def test_summary():
    engine = ExecutionEngine(initial_capital=100)

    engine.open_trade(
        symbol="BTC-USDT",
        direction="BUY",
        entry_price=100,
        stop_loss=98,
        take_profit=104,
        risk_percent=1,
        timestamp=1000,
    )

    engine.close_trade(
        "BTC-USDT",
        104,
        2000,
        "TP",
    )

    summary = engine.get_summary()

    assert summary["closed_trades"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["final_equity"] == 102.0

    print("PASS: trade summary and equity reporting")


def main():
    test_position_size()
    test_buy_take_profit()
    test_sell_stop_loss()
    test_sl_tp_same_candle()
    test_one_position_per_symbol()
    test_dynamic_equity_risk()
    test_summary()

    print("ALL EXECUTION ENGINE TESTS PASSED")


if __name__ == "__main__":
    main()
