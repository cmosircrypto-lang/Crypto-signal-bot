from decimal import Decimal
from io import BytesIO
from unittest.mock import patch

from okx_contract_specs import (
    ContractSpec,
    OKXContractSpecs
)


def make_spec(
    contract_value="0.01",
    lot_size="0.1",
    min_size="0.1"
):
    return ContractSpec(
        inst_id="BTC-USDT-SWAP",
        base_ccy="BTC",
        quote_ccy="USDT",
        settle_ccy="USDT",
        contract_type="linear",
        contract_value=Decimal(contract_value),
        contract_value_currency="BTC",
        lot_size=Decimal(lot_size),
        min_size=Decimal(min_size),
        tick_size=Decimal("0.1"),
        state="live"
    )


def test_contract_parsing():
    loader = OKXContractSpecs()

    item = {
        "instId": "BTC-USDT-SWAP",
        "instType": "SWAP",
        "baseCcy": "BTC",
        "quoteCcy": "USDT",
        "settleCcy": "USDT",
        "ctType": "linear",
        "ctVal": "0.01",
        "ctValCcy": "BTC",
        "lotSz": "0.1",
        "minSz": "0.1",
        "tickSz": "0.1",
        "state": "live"
    }

    spec = loader.parse_instrument(item)

    assert spec is not None
    assert spec.inst_id == "BTC-USDT-SWAP"
    assert spec.contract_value == Decimal("0.01")
    assert spec.lot_size == Decimal("0.1")

    print("PASS: contract specification parsing")


def test_unsupported_contracts_rejected():
    loader = OKXContractSpecs()

    item = {
        "instId": "BTC-USDT-SWAP",
        "instType": "SWAP",
        "baseCcy": "BTC",
        "quoteCcy": "USDT",
        "settleCcy": "USDT",
        "ctType": "inverse",
        "ctVal": "100",
        "ctValCcy": "USD",
        "lotSz": "1",
        "minSz": "1",
        "tickSz": "0.1",
        "state": "live"
    }

    assert loader.parse_instrument(item) is None

    print("PASS: unsupported contracts rejected")


def test_base_quantity_conversion():
    spec = make_spec()

    contracts = spec.base_quantity_to_contracts("0.05")
    assert contracts == Decimal("5")

    quantity = spec.contracts_to_base_quantity("5")
    assert quantity == Decimal("0.05")

    print("PASS: base quantity and contract conversion")


def test_contract_rounding():
    spec = make_spec()

    result = spec.round_contracts_down("1.29")

    assert result == Decimal("1.2")

    print("PASS: contract quantity rounded down")


def test_minimum_order_size():
    spec = make_spec()

    assert spec.validate_order_size("0.1")
    assert not spec.validate_order_size("0.05")
    assert not spec.validate_order_size("0.15")

    print("PASS: minimum and lot-size validation")


def test_price_rounding_down():
    spec = make_spec()

    assert spec.round_price_down("50000.04") == Decimal("50000.0")
    assert spec.round_price_down("50000.10") == Decimal("50000.1")

    print("PASS: price rounded down to tickSz")


def test_price_rounding_up():
    spec = make_spec()

    assert spec.round_price_up("50000.04") == Decimal("50000.1")
    assert spec.round_price_up("50000.10") == Decimal("50000.1")

    print("PASS: price rounded up to tickSz")


def test_buy_exit_price_rounding():
    spec = make_spec()

    stop_loss, take_profit = spec.round_exit_prices(
        "BUY",
        "49000.04",
        "52000.04",
    )

    assert stop_loss == Decimal("49000.0")
    assert take_profit == Decimal("52000.1")

    print("PASS: BUY SL down and TP up")


def test_sell_exit_price_rounding():
    spec = make_spec()

    stop_loss, take_profit = spec.round_exit_prices(
        "SELL",
        "51000.04",
        "48000.04",
    )

    assert stop_loss == Decimal("51000.1")
    assert take_profit == Decimal("48000.0")

    print("PASS: SELL SL up and TP down")


def test_invalid_rounding_direction():
    spec = make_spec()

    try:
        spec.round_exit_prices(
            "SIDEWAYS",
            "49000",
            "52000",
        )
        raise AssertionError(
            "Invalid direction was accepted"
        )
    except ValueError:
        pass

    print("PASS: invalid rounding direction rejected")


def test_risk_based_contract_calculation():
    spec = make_spec()

    contracts = spec.contracts_for_risk(
        entry_price="50000",
        stop_loss="49000",
        risk_amount="10"
    )

    assert contracts == Decimal("1")

    print("PASS: risk-based contract calculation")


def test_risk_rounding_down():
    spec = make_spec()

    contracts = spec.contracts_for_risk(
        entry_price="50000",
        stop_loss="49000",
        risk_amount="15"
    )

    assert contracts == Decimal("1.5")

    print("PASS: risk-based quantity respects lot size")


def test_below_minimum_returns_zero():
    spec = make_spec()

    contracts = spec.contracts_for_risk(
        entry_price="50000",
        stop_loss="49000",
        risk_amount="0.5"
    )

    assert contracts == Decimal("0")

    print("PASS: below-minimum trade rejected")


def test_invalid_risk_inputs():
    spec = make_spec()

    try:
        spec.contracts_for_risk(
            entry_price="50000",
            stop_loss="50000",
            risk_amount="10"
        )
        raise AssertionError("Equal prices were accepted")
    except ValueError:
        pass

    try:
        spec.contracts_for_risk(
            entry_price="50000",
            stop_loss="49000",
            risk_amount="0"
        )
        raise AssertionError("Zero risk was accepted")
    except ValueError:
        pass

    print("PASS: invalid risk inputs rejected")


def test_mocked_api_loading():
    loader = OKXContractSpecs()

    valid_item = {
        "instId": "BTC-USDT-SWAP",
        "instType": "SWAP",
        "baseCcy": "BTC",
        "quoteCcy": "USDT",
        "settleCcy": "USDT",
        "ctType": "linear",
        "ctVal": "0.01",
        "ctValCcy": "BTC",
        "lotSz": "0.1",
        "minSz": "0.1",
        "tickSz": "0.1",
        "state": "live"
    }

    payload = {
        "code": "0",
        "data": [valid_item]
    }

    class FakeResponse(BytesIO):

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()
            return False

    def fake_urlopen(url, timeout):
        assert "instType=SWAP" in url
        assert timeout == 20

        return FakeResponse(
            __import__("json").dumps(payload).encode("utf-8")
        )

    with patch(
        "okx_contract_specs.urlopen",
        side_effect=fake_urlopen
    ):
        contracts = loader.load()

    assert "BTC-USDT-SWAP" in contracts
    assert loader.get("BTC-USDT-SWAP").base_ccy == "BTC"

    print("PASS: mocked OKX API loading")


if __name__ == "__main__":
    test_contract_parsing()
    test_unsupported_contracts_rejected()
    test_base_quantity_conversion()
    test_contract_rounding()
    test_minimum_order_size()

    test_price_rounding_down()
    test_price_rounding_up()
    test_buy_exit_price_rounding()
    test_sell_exit_price_rounding()
    test_invalid_rounding_direction()

    test_risk_based_contract_calculation()
    test_risk_rounding_down()
    test_below_minimum_returns_zero()
    test_invalid_risk_inputs()
    test_mocked_api_loading()

    print("ALL OKX CONTRACT SPECS TESTS PASSED")