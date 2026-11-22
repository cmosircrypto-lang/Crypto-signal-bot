"""CLI for the frozen genetic champion forward-only validation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from .forward_evaluator import run_forward_validation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Backtest/data/okx")
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="Backtest/results/genetic_frozen_forward")
    parser.add_argument("--expected-symbols", type=int, default=76)
    args = parser.parse_args(argv)
    if args.expected_symbols != 76:
        parser.error("this frozen validation is locked to the validated 76-symbol dataset")

    root = Path.cwd()
    sys.path.insert(0, str(root / "Backtest/SwingEngine"))
    from data_loader import load_okx_data

    quality = json.loads(Path(args.quality_report).read_text(encoding="utf-8"))
    print("[LOAD] loading and validating the 76-symbol OKX dataset", flush=True)
    data = load_okx_data(args.data_root, expected_symbols=76)
    print("[VALIDATE] checking frozen candidates and exact forward window", flush=True)
    report = run_forward_validation(data, quality, args.manifest)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "frozen_forward_validation.json"
    csv_path = output_dir / "frozen_forward_validation.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = [
        "candidate_id", "symbol", "direction", "trade_count", "net_pnl",
        "average_r", "profit_factor", "max_drawdown_pct", "gross_profit", "gross_loss",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["results"]:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "symbol": row["symbol"],
                "direction": row["direction"],
                **row["forward_metrics"],
            })

    print(f"[COMPLETE] candidates={report['candidate_count']}; evolution_ran=False", flush=True)
    print(f"[REPORT] {json_path}", flush=True)
    print(f"[REPORT] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
