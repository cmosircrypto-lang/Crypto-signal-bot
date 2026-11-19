from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import requests


OKX_PUBLIC_URL = "https://www.okx.com/api/v5/public/instruments"


def to_decimal(value, field_name):
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(
            f"Invalid {field_name}: {value}"
        ) from exc

    if not result.is_finite() or result <= 0:
        raise ValueError(f"Invalid {field_name}: {value}")

    return result


@dataclass(frozen=True)
class ContractSpec:
    inst_id: str
    base_ccy: str
    quote_ccy: str
    settle_ccy: str
    contract_type: str
    contract_value: Decimal
    contract_value_currency: str
    lot_size: Decimal
    min_size: Decimal
    tick_size: Decimal
    state: str

    def base_quantity_to_contracts(self, base_quantity):
        """
        Converts base-asset quantity into contract quantity.

        Only supports linear contracts whose contract value
        is expressed in the base currency.
        """

        quantity = to_decimal(
            base_quantity,
            "base_quantity"
        )

        if self.contract_type != "linear":
            raise ValueError("Only linear contracts are supported")

        if self.contract_value_currency != self.base_ccy:
            raise ValueError(
                "Unsupported contract value currency"
            )

        return quantity / self.contract_value

    def contracts_to_base_quantity(self, contracts):
        """
        Converts contract quantity into base-asset quantity.
        """

        count = to_decimal(contracts, "contracts")

        if self.contract_type != "linear":
            raise ValueError("Only linear contracts are supported")

        if self.contract_value_currency != self.base_ccy:
            raise ValueError(
                "Unsupported contract value currency"
            )

        return count * self.contract_value

    def round_contracts_down(self, contracts):
        """
        Rounds contracts down to the allowed lot size.
        Never rounds risk exposure upward.
        """

        count = Decimal(str(contracts))

        if not count.is_finite() or count <= 0:
            return Decimal("0")

        steps = (
            count / self.lot_size
        ).to_integral_value(rounding=ROUND_DOWN)

        return steps * self.lot_size

    def validate_order_size(self, contracts):
        """
        Checks minimum size and lot-size alignment.
        """

        count = Decimal(str(contracts))

        if not count.is_finite() or count <= 0:
            return False

        if count < self.min_size:
            return False

        if count % self.lot_size != 0:
            return False

        return True

    def contracts_for_risk(
        self,
        entry_price,
        stop_loss,
        risk_amount
    ):
        """
        Calculates contract count from a fixed USDT risk.

        Assumptions:
        - Linear USDT-settled contract.
        - Contract value is expressed in base currency.
        - Fees and slippage are not included yet.
        """

        entry = to_decimal(entry_price, "entry_price")
        stop = to_decimal(stop_loss, "stop_loss")
        risk = to_decimal(risk_amount, "risk_amount")

        price_distance = abs(entry - stop)

        if price_distance == 0:
            raise ValueError("Entry and stop-loss cannot be equal")

        if self.contract_type != "linear":
            raise ValueError("Only linear contracts are supported")

        if self.settle_ccy != "USDT":
            raise ValueError("Only USDT-settled contracts are supported")

        if self.contract_value_currency != self.base_ccy:
            raise ValueError(
                "Unsupported contract value currency"
            )

        # Risk per contract = price distance * contract value
        risk_per_contract = (
            price_distance * self.contract_value
        )

        raw_contracts = risk / risk_per_contract

        contracts = self.round_contracts_down(
            raw_contracts
        )

        if not self.validate_order_size(contracts):
            return Decimal("0")

        return contracts


class OKXContractSpecs:

    def __init__(self, timeout=20):
        self.timeout = timeout
        self.contracts = {}

    @staticmethod
    def parse_instrument(item):
        inst_id = item.get("instId", "")

        if not inst_id.endswith("-USDT-SWAP"):
            return None

        if item.get("instType") != "SWAP":
            return None

        if item.get("settleCcy") != "USDT":
            return None

        if item.get("ctType") != "linear":
            return None

        if item.get("state") != "live":
            return None

        base_ccy = item.get("baseCcy", "")

        # For linear USDT contracts, require contract value
        # to be expressed in the base currency.
        if item.get("ctValCcy") != base_ccy:
            return None

        return ContractSpec(
            inst_id=inst_id,
            base_ccy=base_ccy,
            quote_ccy=item.get("quoteCcy", "USDT"),
            settle_ccy=item["settleCcy"],
            contract_type=item["ctType"],
            contract_value=to_decimal(
                item["ctVal"], "ctVal"
            ),
            contract_value_currency=item["ctValCcy"],
            lot_size=to_decimal(
                item["lotSz"], "lotSz"
            ),
            min_size=to_decimal(
                item["minSz"], "minSz"
            ),
            tick_size=to_decimal(
                item["tickSz"], "tickSz"
            ),
            state=item["state"]
        )

    def load(self):
        """
        Downloads public OKX USDT linear SWAP specifications.
        No API key is required.
        """

        response = requests.get(
            OKX_PUBLIC_URL,
            params={"instType": "SWAP"},
            timeout=self.timeout
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("code") != "0":
            raise RuntimeError(
                f"OKX API error: {payload}"
            )

        data = payload.get("data")

        if not isinstance(data, list):
            raise RuntimeError("Invalid OKX instruments response")

        loaded = {}

        for item in data:
            spec = self.parse_instrument(item)

            if spec is not None:
                loaded[spec.inst_id] = spec

        if not loaded:
            raise RuntimeError(
                "No valid USDT linear SWAP contracts found"
            )

        self.contracts = loaded

        return self.contracts

    def get(self, inst_id):
        if inst_id not in self.contracts:
            raise KeyError(
                f"Contract not found: {inst_id}"
            )

        return self.contracts[inst_id]

    def get_all(self):
        return dict(self.contracts)

    def symbols(self):
        return sorted(self.contracts.keys())