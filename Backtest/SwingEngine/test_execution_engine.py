
from decimal import Decimal

from execution_engine import ExecutionEngine
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


def make_contract_engine():
    spec = make_spec()

    return ExecutionEngine(
        initial_capital=100,
        contract_specs={
            "BTC-USDT-SWAP": spec,
        },
    )


# --------------------------------------------------
# Legacy execution tests
# --------------------------------------------------

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
    assert abs(engine.equity - 102.0) < 1e-9

    print("PASS: BUY take-profit execution")


def test_sell_stop_loss():
    engine = ExecutionEngine(initial_capital=100)

    engine.open_trade(
        symbol="BTC-USDT",
        direction="SELL",
        entry_price=100,
        stop_loss=102,
        take_profit=96,
        risk_percent=1,
        timestamp=1000,
    )

    result = engine.check_exit(
        "BTC-USDT",
        {"high": 103, "low": 99},
        2000,
    )

    assert result is not None
    assert result.exit_reason == "SL"
    assert abs(result.pnl + 1.0) < 1e-9
    assert abs(engine.equity - 99.0) < 1e-9

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
        raise AssertionError(
            "Duplicate symbol position was accepted"
        )
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
    assert abs(summary["final_equity"] - 102.0) < 1e-9

    print("PASS: trade summary and equity reporting")


# --------------------------------------------------
# OKX contract-aware execution tests
# --------------------------------------------------

def test_contract_based_position_size():
    engine = make_contract_engine()

    contracts, quantity, actual_risk = (
        engine.calculate_contract_position_size(
            symbol="BTC-USDT-SWAP",
            entry_price=50000,
            stop_loss=49000,
            risk_percent=1,
        )
    )

    assert abs(contracts - 0.1) < 1e-9
    assert abs(quantity - 0.001) < 1e-9
    assert abs(actual_risk - 1.0) < 1e-9

    print("PASS: OKX contract-based position sizing")


def test_contract_rounding_keeps_risk_below_target():
    engine = make_contract_engine()

    contracts, quantity, actual_risk = (
        engine.calculate_contract_position_size(
            symbol="BTC-USDT-SWAP",
            entry_price=50000,
            stop_loss=49000,
            risk_percent=1.5,
        )
    )

    requested_risk = 1.5

    assert abs(contracts - 0.1) < 1e-9
    assert abs(quantity - 0.001) < 1e-9
    assert actual_risk <= requested_risk
    assert abs(actual_risk - 1.0) < 1e-9

    print("PASS: contract lot rounding does not exceed risk")


def test_recovery_rounds_contracts_up_within_explicit_cap():
    engine = make_contract_engine()

    position = engine.open_trade(
        symbol="BTC-USDT-SWAP",
        direction="BUY",
        entry_price=50000,
        stop_loss=49000,
        take_profit=52000,
        risk_percent=1.5,
        timestamp=1000,
        round_contracts_up=True,
        max_risk_amount=2.0,
    )

    assert abs(position.contract_count - 0.2) < 1e-9
    assert abs(position.quantity - 0.002) < 1e-9
    assert abs(position.risk_amount - 2.0) < 1e-9

    print("PASS: recovery lot rounding up stays within explicit risk cap")


def test_recovery_rounding_up_respects_minimum_contract_size():
    spec = make_spec(lot_size="0.1", min_size="0.5")
    engine = ExecutionEngine(
        initial_capital=100,
        contract_specs={"BTC-USDT-SWAP": spec},
    )

    position = engine.open_trade(
        symbol="BTC-USDT-SWAP",
        direction="BUY",
        entry_price=50000,
        stop_loss=49000,
        take_profit=52000,
        risk_percent=1.5,
        timestamp=1000,
        round_contracts_up=True,
        max_risk_amount=5.0,
    )

    assert abs(position.contract_count - 0.5) < 1e-9
    assert abs(position.risk_amount - 5.0) < 1e-9
    print("PASS: recovery upward rounding honors minSz within risk cap")


def test_recovery_rounding_up_rejected_above_explicit_cap():
    engine = make_contract_engine()

    try:
        engine.open_trade(
            symbol="BTC-USDT-SWAP",
            direction="BUY",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            risk_percent=1.5,
            timestamp=1000,
            round_contracts_up=True,
            max_risk_amount=1.5,
        )
        raise AssertionError("Up-rounded risk exceeded the supplied cap")
    except ValueError as exc:
        assert "risk budget" in str(exc).lower()

    assert "BTC-USDT-SWAP" not in engine.open_positions
    print("PASS: recovery lot rounding up cannot exceed explicit risk cap")


def test_below_minimum_contract_size_rejected():
    engine = make_contract_engine()

    try:
        engine.open_trade(
            symbol="BTC-USDT-SWAP",
            direction="BUY",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            risk_percent=0.5,
            timestamp=1000,
        )
        raise AssertionError(
            "Trade below minSz was accepted"
        )
    except ValueError:
        pass

    assert "BTC-USDT-SWAP" not in engine.open_positions

    print("PASS: below-minimum contract size rejected")


def test_contract_pnl_uses_base_quantity():
    engine = make_contract_engine()

    position = engine.open_trade(
        symbol="BTC-USDT-SWAP",
        direction="BUY",
        entry_price=50000,
        stop_loss=49000,
        take_profit=52000,
        risk_percent=1,
        timestamp=1000,
    )

    assert abs(position.contract_count - 0.1) < 1e-9
    assert abs(position.quantity - 0.001) < 1e-9
    assert abs(position.risk_amount - 1.0) < 1e-9

    result = engine.check_exit(
        "BTC-USDT-SWAP",
        {"high": 52100, "low": 50100},
        2000,
    )

    assert result is not None
    assert result.exit_reason == "TP"
    assert abs(result.contract_count - 0.1) < 1e-9
    assert abs(result.quantity - 0.001) < 1e-9
    assert abs(result.pnl - 2.0) < 1e-9
    assert abs(engine.equity - 102.0) < 1e-9

    print("PASS: contract PnL uses actual base quantity")


def test_missing_contract_spec_rejected():
    engine = make_contract_engine()

    try:
        engine.open_trade(
            symbol="ETH-USDT-SWAP",
            direction="BUY",
            entry_price=3000,
            stop_loss=2900,
            take_profit=3200,
            risk_percent=1,
            timestamp=1000,
        )
        raise AssertionError(
            "Trade without contract specification was accepted"
        )
    except KeyError:
        pass

    assert "ETH-USDT-SWAP" not in engine.open_positions

    print("PASS: missing contract specification rejected")


def test_contract_sell_pnl():
    engine = make_contract_engine()

    position = engine.open_trade(
        symbol="BTC-USDT-SWAP",
        direction="SELL",
        entry_price=50000,
        stop_loss=51000,
        take_profit=48000,
        risk_percent=1,
        timestamp=1000,
    )

    assert abs(position.contract_count - 0.1) < 1e-9
    assert abs(position.quantity - 0.001) < 1e-9

    result = engine.check_exit(
        "BTC-USDT-SWAP",
        {"high": 49900, "low": 47900},
        2000,
    )

    assert result is not None
    assert result.exit_reason == "TP"
    assert abs(result.pnl - 2.0) < 1e-9
    assert abs(engine.equity - 102.0) < 1e-9

    print("PASS: contract-based SELL PnL calculation")


# --------------------------------------------------
# Tick-size integration tests
# --------------------------------------------------

def test_buy_tick_rounding_and_risk_recalculation():
    spec = make_spec()

    engine = ExecutionEngine(
        initial_capital=999.95,
        contract_specs={
            "BTC-USDT-SWAP": spec,
        },
    )

    position = engine.open_trade(
        symbol="BTC-USDT-SWAP",
        direction="BUY",
        entry_price=50000.0,
        stop_loss=49000.06,
        take_profit=52000.04,
        risk_percent=1,
        timestamp=1000,
    )

    assert position.stop_loss == 49000.0
    assert position.take_profit == 52000.1
    assert abs(position.contract_count - 0.9) < 1e-9
    assert abs(position.quantity - 0.009) < 1e-9
    assert position.risk_amount <= 9.9995

    print("PASS: BUY tick rounding and risk recalculation")


def test_sell_tick_rounding_and_risk_recalculation():
    spec = make_spec()

    engine = ExecutionEngine(
        initial_capital=1000.04,
        contract_specs={
            "BTC-USDT-SWAP": spec,
        },
    )

    position = engine.open_trade(
        symbol="BTC-USDT-SWAP",
        direction="SELL",
        entry_price=50000.0,
        stop_loss=51000.04,
        take_profit=48000.04,
        risk_percent=1,
        timestamp=1000,
    )

    assert position.stop_loss == 51000.1
    assert position.take_profit == 48000.0
    assert abs(position.contract_count - 0.9) < 1e-9
    assert abs(position.quantity - 0.009) < 1e-9
    assert position.risk_amount <= 10.0004

    print("PASS: SELL tick rounding and risk recalculation")


def test_tick_rounding_preserves_valid_price_structure():
    spec = make_spec()

    engine = ExecutionEngine(
        initial_capital=100,
        contract_specs={
            "BTC-USDT-SWAP": spec,
        },
    )

    try:
        engine.open_trade(
            symbol="BTC-USDT-SWAP",
            direction="BUY",
            entry_price=50000.0,
            stop_loss=50000.04,
            take_profit=50001.0,
            risk_percent=1,
            timestamp=1000,
        )

        raise AssertionError(
            "Invalid rounded BUY structure was accepted"
        )

    except ValueError:
        pass

    assert "BTC-USDT-SWAP" not in engine.open_positions

    print("PASS: invalid price structure after rounding rejected")


# --------------------------------------------------
# Test runner
# --------------------------------------------------

def main():
    test_position_size()
    test_buy_take_profit()
    test_sell_stop_loss()
    test_sl_tp_same_candle()
    test_one_position_per_symbol()
    test_dynamic_equity_risk()
    test_summary()

    test_contract_based_position_size()
    test_contract_rounding_keeps_risk_below_target()
    test_recovery_rounds_contracts_up_within_explicit_cap()
    test_recovery_rounding_up_respects_minimum_contract_size()
    test_recovery_rounding_up_rejected_above_explicit_cap()
    test_below_minimum_contract_size_rejected()
    test_contract_pnl_uses_base_quantity()
    test_missing_contract_spec_rejected()
    test_contract_sell_pnl()

    test_buy_tick_rounding_and_risk_recalculation()
    test_sell_tick_rounding_and_risk_recalculation()
    test_tick_rounding_preserves_valid_price_structure()

    print("ALL EXECUTION ENGINE TESTS PASSED")


if __name__ == "__main__":
    main()
