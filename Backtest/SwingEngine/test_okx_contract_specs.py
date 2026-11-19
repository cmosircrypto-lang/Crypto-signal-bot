from decimal import Decimal

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


def test_risk_based_contract_calculation():
    spec = make_spec()

    # Contract value = 0.01 BTC
    # Entry = 50000 USDT
    # Stop = 49000 USDT
    # Risk per contract = 1000 * 0.01 = 10 USDT
    # Risk budget = 10 USDT
    # Expected quantity = 1 contract

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

    # Raw size = 1.5 contracts.
    # Lot size = 0.1, so 1.5 remains valid.

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
        assert False
    except ValueError:
        pass

    try:
        spec.contracts_for_risk(
            entry_price="50000",
            stop_loss="49000",
            risk_amount="0"
        )
        assert False
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

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "code": "0",
                "data": [valid_item]
            }

    def fake_get(*args, **kwargs):
        return FakeResponse()

    loader_module = __import__(
        "okx_contract_specs"
    )

    original_get = loader_module.requests.get
    loader_module.requests.get = fake_get

    try:
        contracts = loader.load()

        assert "BTC-USDT-SWAP" in contracts
        assert loader.get("BTC-USDT-SWAP").base_ccy == "BTC"

    finally:
        loader_module.requests.get = original_get

    print("PASS: mocked OKX API loading")


if __name__ == "__main__":
    test_contract_parsing()
    test_unsupported_contracts_rejected()
    test_base_quantity_conversion()
    test_contract_rounding()
    test_minimum_order_size()
    test_risk_based_contract_calculation()
    test_risk_rounding_down()
    test_below_minimum_returns_zero()
    test_invalid_risk_inputs()
    test_mocked_api_loading()

    print("ALL OKX CONTRACT SPECS TESTS PASSED")