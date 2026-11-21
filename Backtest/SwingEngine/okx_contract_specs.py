from dataclasses import dataclass
from decimal import (
    Decimal,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_CEILING,
)
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json


OKX_PUBLIC_URL = "https://openapi.okx.com/api/v5/public/instruments"


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

    def round_price_down(self, price):
        """
        Rounds a positive price down to the nearest tick.
        """
        value = to_decimal(price, "price")

        steps = (
            value / self.tick_size
        ).to_integral_value(
            rounding=ROUND_FLOOR
        )

        return steps * self.tick_size

    def round_price_up(self, price):
        """
        Rounds a positive price up to the nearest tick.
        """
        value = to_decimal(price, "price")

        steps = (
            value / self.tick_size
        ).to_integral_value(
            rounding=ROUND_CEILING
        )

        return steps * self.tick_size

    def round_exit_prices(
        self,
        direction,
        stop_loss,
        take_profit,
    ):
        """
        Rounds SL and TP outward so rounding does not
        move either exit closer to the entry.

        BUY:
            SL down
            TP up

        SELL:
            SL up
            TP down
        """

        if not isinstance(direction, str):
            raise ValueError("Invalid direction")

        direction = direction.strip().upper()

        if direction == "BUY":
            rounded_sl = self.round_price_down(stop_loss)
            rounded_tp = self.round_price_up(take_profit)

        elif direction == "SELL":
            rounded_sl = self.round_price_up(stop_loss)
            rounded_tp = self.round_price_down(take_profit)

        else:
            raise ValueError("Direction must be BUY or SELL")

        return rounded_sl, rounded_tp

    def base_quantity_to_contracts(self, base_quantity):
        quantity = to_decimal(
            base_quantity,
            "base_quantity"
        )

        if self.contract_type != "linear":
            raise ValueError(
                "Only linear contracts are supported"
            )

        if self.contract_value_currency != self.base_ccy:
            raise ValueError(
                "Unsupported contract value currency"
            )

        return quantity / self.contract_value

    def contracts_to_base_quantity(self, contracts):
        count = to_decimal(
            contracts,
            "contracts"
        )

        if self.contract_type != "linear":
            raise ValueError(
                "Only linear contracts are supported"
            )

        if self.contract_value_currency != self.base_ccy:
            raise ValueError(
                "Unsupported contract value currency"
            )

        return count * self.contract_value

    def round_contracts_down(self, contracts):
        count = Decimal(str(contracts))

        if not count.is_finite() or count <= 0:
            return Decimal("0")

        steps = (
            count / self.lot_size
        ).to_integral_value(
            rounding=ROUND_DOWN
        )

        return steps * self.lot_size

    def round_contracts_up(self, contracts):
        """Round a positive contract count up to the next lotSz increment."""
        count = Decimal(str(contracts))
        if not count.is_finite() or count <= 0:
            return Decimal("0")
        steps = (count / self.lot_size).to_integral_value(
            rounding=ROUND_CEILING
        )
        return steps * self.lot_size

    def validate_order_size(self, contracts):
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
        risk_amount,
        round_up=False,
    ):
        entry = to_decimal(
            entry_price,
            "entry_price"
        )

        stop = to_decimal(
            stop_loss,
            "stop_loss"
        )

        risk = to_decimal(
            risk_amount,
            "risk_amount"
        )

        price_distance = abs(entry - stop)

        if price_distance == 0:
            raise ValueError(
                "Entry and stop-loss cannot be equal"
            )

        if self.contract_type != "linear":
            raise ValueError(
                "Only linear contracts are supported"
            )

        if self.settle_ccy != "USDT":
            raise ValueError(
                "Only USDT-settled contracts are supported"
            )

        if self.contract_value_currency != self.base_ccy:
            raise ValueError(
                "Unsupported contract value currency"
            )

        risk_per_contract = (
            price_distance * self.contract_value
        )

        raw_contracts = (
            risk / risk_per_contract
        )

        contracts = (
            self.round_contracts_up(raw_contracts)
            if round_up
            else self.round_contracts_down(raw_contracts)
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

        # OKX documents baseCcy/quoteCcy for SPOT and MARGIN;
        # these fields may be empty for SWAP instruments. Derive the
        # currencies from the canonical instrument ID instead.
        parts = inst_id.split("-")
        if len(parts) != 3 or parts[1] != "USDT" or parts[2] != "SWAP":
            return None

        base_ccy = parts[0]
        quote_ccy = parts[1]
        contract_value_currency = item.get("ctValCcy", "")

        if not base_ccy or contract_value_currency != base_ccy:
            return None

        return ContractSpec(
            inst_id=inst_id,
            base_ccy=base_ccy,
            quote_ccy=quote_ccy,
            settle_ccy=item["settleCcy"],
            contract_type=item["ctType"],
            contract_value=to_decimal(
                item["ctVal"],
                "ctVal"
            ),
            contract_value_currency=contract_value_currency,
            lot_size=to_decimal(
                item["lotSz"],
                "lotSz"
            ),
            min_size=to_decimal(
                item["minSz"],
                "minSz"
            ),
            tick_size=to_decimal(
                item["tickSz"],
                "tickSz"
            ),
            state=item["state"]
        )

    def load(self):
        """
        Downloads public OKX USDT linear SWAP specifications.
        No API key or external Python package is required.
        """

        params = urlencode({
            "instType": "SWAP"
        })

        url = f"{OKX_PUBLIC_URL}?{params}"

        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "SwingEngineBacktest/1.0",
                    "Accept": "application/json",
                },
            )
            with urlopen(
                request,
                timeout=self.timeout
            ) as response:

                payload = json.loads(
                    response.read().decode("utf-8")
                )

        except Exception as exc:
            raise RuntimeError(
                "Failed to load OKX contract specifications: "
                f"{exc}"
            ) from exc

        if payload.get("code") != "0":
            raise RuntimeError(
                f"OKX API error: {payload}"
            )

        data = payload.get("data")

        if not isinstance(data, list):
            raise RuntimeError(
                "Invalid OKX instruments response"
            )

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
        return sorted(
            self.contracts.keys()
        )