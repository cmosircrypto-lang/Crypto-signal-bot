"""Smoke test for the OKX data loader and agreed portfolio constraints."""
from pathlib import Path

from data_loader import dataset_report, load_okx_data, latest_closed_index
from engine import PortfolioManager, PortfolioLimitError


def test_portfolio_rules():
    portfolio = PortfolioManager()

    portfolio.open_trade("BTC-USDT-SWAP", "BUY", "t1")
    assert not portfolio.should_scan("BTC-USDT-SWAP")
    assert portfolio.can_open("BTC-USDT-SWAP", "SELL") == (
        False, "SYMBOL_ALREADY_OPEN"
    )

    portfolio.open_trade("ETH-USDT-SWAP", "BUY", "t2")
    portfolio.open_trade("SOL-USDT-SWAP", "BUY", "t3")
    assert portfolio.can_open("DOGE-USDT-SWAP", "BUY") == (False, "BUY_LIMIT")

    portfolio.open_trade("XRP-USDT-SWAP", "SELL", "t4")
    portfolio.open_trade("ADA-USDT-SWAP", "SELL", "t5")
    portfolio.open_trade("LINK-USDT-SWAP", "SELL", "t6")
    assert portfolio.snapshot()["open_total"] == 6
    assert portfolio.can_open("AVAX-USDT-SWAP", "SELL") == (
        False, "PORTFOLIO_TOTAL_LIMIT"
    )

    portfolio.close_trade("BTC-USDT-SWAP", "t1")
    assert portfolio.should_scan("BTC-USDT-SWAP")
    assert portfolio.can_open("BTC-USDT-SWAP", "BUY") == (True, "OK")

    try:
        portfolio.open_trade("BTC-USDT-SWAP", "BUY", "t7")
    except PortfolioLimitError:
        raise AssertionError("A freed BUY slot should be available")
    assert portfolio.snapshot()["open_total"] == 6


def main():
    test_portfolio_rules()
    print("PASS: portfolio constraints (6 total, 3 BUY, 3 SELL, one per symbol)")

    root = Path("Backtest/data/okx")
    data = load_okx_data(root, expected_symbols=76)
    report = dataset_report(data)
    print(f"PASS: loaded {len(data)} symbols across 15m/1h/4h")
    print(report.groupby("timeframe")["candles"].agg(["count", "sum"]).to_string())

    sample = next(iter(data.values())).frames["4h"]
    last_open = int(sample["ts"].iloc[-1])
    last_close = last_open + 4 * 60 * 60 * 1000
    before_last_close = latest_closed_index(sample, "4h", last_close - 1)
    assert before_last_close is None or before_last_close < len(sample) - 1
    assert latest_closed_index(sample, "4h", last_close) == len(sample) - 1
    print("PASS: higher-timeframe candles are exposed only after close")


if __name__ == "__main__":
    main()
