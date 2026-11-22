from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from Backtest.GeneticEngine.freeze_candidates import build_frozen_manifest
from Backtest.GeneticEngine.forward_evaluator import (
    run_forward_validation, validate_quality_report,
)
from Backtest.GeneticEngine.genome import StrategyGenome
from Backtest.GeneticEngine.historical_evaluator import FoldResult

SYMBOLS = (
    "ADA-USDT-SWAP", "BNB-USDT-SWAP", "BTC-USDT-SWAP",
    "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP",
)
START = 1_794_873_600_000  # 2026-11-17T00:00:00Z
END = 1_795_384_800_000    # 2026-11-22T22:00:00Z


def quality_report(names):
    return {
        "status": "PASS",
        "errors": [],
        "forward_window_start_utc": "2026-11-17T00:00:00+00:00",
        "validated_through_exclusive_utc": "2026-11-22T22:00:00+00:00",
        "bar_interval_minutes": 15,
        "symbols": [{
            "symbol": name,
            "status": "OK",
            "new_rows": 568,
            "new_window_missing": 0,
            "new_window_extra": 0,
            "new_first_utc": "2026-11-17T00:00:00+00:00",
            "new_last_utc": "2026-11-22T21:45:00+00:00",
        } for name in names],
    }


def frames_for_window():
    return {
        "15m": pd.DataFrame({"ts": range(START, END, 900_000),
                             "open": 100.0, "high": 101.0, "low": 99.0,
                             "close": 100.0, "volume": 1.0}),
        "1h": pd.DataFrame({"ts": range(START, END, 3_600_000),
                            "open": 100.0, "high": 101.0, "low": 99.0,
                            "close": 100.0, "volume": 1.0}),
        "4h": pd.DataFrame({"ts": range(START, END - 14_400_000, 14_400_000),
                            "open": 100.0, "high": 101.0, "low": 99.0,
                            "close": 100.0, "volume": 1.0}),
    }


class FrozenForwardValidationTests(unittest.TestCase):
    def test_quality_report_requires_complete_exact_window(self):
        names = [f"SYMBOL-{i:02d}" for i in range(76)]
        report = quality_report(names)
        start_ms, end_ms = validate_quality_report(report)
        self.assertEqual(start_ms, START)
        self.assertEqual(end_ms, END)
        report["symbols"][0]["new_rows"] = 567
        with self.assertRaises(ValueError):
            validate_quality_report(report)

    def test_freezer_copies_exactly_eleven_eligible_genomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for symbol in SYMBOLS:
                for direction in ("LONG", "SHORT"):
                    eligible = not (symbol == "BNB-USDT-SWAP" and direction == "LONG")
                    genome = StrategyGenome(symbols=(symbol,)).to_dict()
                    summary = {
                        "symbol": symbol,
                        "direction": direction,
                        "result": {
                            "champion": {
                                "eligible": eligible,
                                "genome": genome,
                                "metrics": {"trade_count": 10, "profit_factor": 1.2},
                            }
                        },
                    }
                    path = root / f"genetic_family_summary_{symbol}_{direction}.json"
                    path.write_text(json.dumps(summary), encoding="utf-8")
            frozen = build_frozen_manifest(
                root, source_run_id=45054615629,
                source_commit_sha="16544a09a07ce024f07037cb0ca65c1fbd18cf57",
            )
            self.assertEqual(len(frozen["candidates"]), 11)
            self.assertEqual(frozen["excluded"][0]["candidate_id"], "BNB-USDT-SWAP__LONG")
            self.assertTrue(frozen["frozen"])

    def test_forward_runner_calls_each_frozen_candidate_once(self):
        names = list(SYMBOLS) + [f"OTHER-{i:02d}" for i in range(70)]
        data = {name: frames_for_window() for name in names}
        candidates = []
        for symbol in SYMBOLS:
            for direction in ("LONG", "SHORT"):
                if symbol == "BNB-USDT-SWAP" and direction == "LONG":
                    continue
                candidates.append({
                    "candidate_id": f"{symbol}__{direction}",
                    "symbol": symbol,
                    "direction": direction,
                    "genome": StrategyGenome(symbols=(symbol,)).to_dict(),
                    "historical_reference_metrics": {},
                })
        manifest = {
            "schema_version": 1,
            "frozen": True,
            "source": {"run_id": 45054615629, "commit_sha": "source"},
            "window": {
                "start_utc": "2026-11-17T00:00:00Z",
                "end_exclusive_utc": "2026-11-22T22:00:00Z",
            },
            "candidates": candidates,
            "excluded": [{"candidate_id": "BNB-USDT-SWAP__LONG"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "frozen.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            fake_result = FoldResult(
                start_ms=START, end_ms=END, trade_count=3, net_pnl=1.5,
                average_r=0.5, profit_factor=1.5, max_drawdown_pct=0.8,
                gross_profit=3.0, gross_loss=2.0,
            )
            with patch(
                "Backtest.GeneticEngine.forward_evaluator._simulate_fold",
                return_value=fake_result,
            ) as simulate:
                report = run_forward_validation(data, quality_report(names), manifest_path)
            self.assertEqual(report["candidate_count"], 11)
            self.assertFalse(report["evolution_ran"])
            self.assertFalse(report["candidate_parameters_changed"])
            self.assertEqual(simulate.call_count, 11)
            self.assertEqual(simulate.call_args.kwargs["initial_equity"], 100.0)


if __name__ == "__main__":
    unittest.main()
