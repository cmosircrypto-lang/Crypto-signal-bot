"""Run one symbol/direction while preserving compatible historical checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

from . import config
from .genetic_runner import run_direction_evolution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Backtest/data/okx")
    parser.add_argument("--checkpoint-dir", default="Backtest/results/genetic_family_checkpoints")
    parser.add_argument("--output", default="Backtest/results/genetic_family_direction_summary.json")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--direction", choices=("LONG", "SHORT"), required=True)
    parser.add_argument("--profile", choices=("smoke", "main"), default="smoke")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--expected-symbols", type=int, default=76)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--oos-fraction", type=float, default=0.15)
    parser.add_argument("--warmup-fraction", type=float, default=0.50)
    args = parser.parse_args(argv)
    if args.population < 2 or args.generations < 1:
        parser.error("population must be >=2 and generations >=1")

    root = Path.cwd()
    sys.path.insert(0, str(root / "Backtest/SwingEngine"))
    from data_loader import load_okx_data
    from .run_genetic_research import _dataset_hash, _source_hash
    from .historical_evaluator import evaluate_candidate_walk_forward

    symbol = args.symbol.strip().upper()
    direction = args.direction.upper()
    started = time.monotonic()
    print(f"[LOAD] validating cached dataset at {args.data_root}", flush=True)
    data = load_okx_data(args.data_root, expected_symbols=args.expected_symbols)
    if symbol not in data:
        parser.error(f"unknown symbol: {symbol}")

    if args.profile == "smoke":
        evolution = config.EvolutionConfig(
            random_seed=args.seed,
            smoke_population=args.population,
            smoke_generations=args.generations,
            tournament_size=2,
            crossover_probability=0.8,
            mutation_probability=0.2,
        )
    else:
        evolution = config.EvolutionConfig(random_seed=args.seed)
    evolution.validate()

    print(f"[HASH] hashing {len(data)} symbols and research source", flush=True)
    metadata = {
        "code_hash": _source_hash(root),
        "config_hash": hashlib.sha256(json.dumps({
            "evolution": asdict(evolution),
            "profile": args.profile,
            "fold_count": args.fold_count,
            "oos_fraction": args.oos_fraction,
            "warmup_fraction": args.warmup_fraction,
            "fees": 0.0,
            "slippage": 0.0,
            "same_candle_policy": "SL_FIRST",
        }, sort_keys=True).encode()).hexdigest(),
        "dataset_hash": _dataset_hash(data),
    }
    metadata["walk_forward"] = {
        "oos_fraction": args.oos_fraction,
        "warmup_fraction": args.warmup_fraction,
        "fold_count": args.fold_count,
    }
    metadata["swing_signal_provider"] = None

    def evaluator(spec):
        from .genome import StrategyGenome
        genome = StrategyGenome.from_dict(spec["genome"])
        return evaluate_candidate_walk_forward(
            data, symbol, genome, direction,
            oos_fraction=args.oos_fraction,
            warmup_fraction=args.warmup_fraction,
            fold_count=args.fold_count,
            swing_signal_provider=None,
        )

    checkpoint = Path(args.checkpoint_dir) / f"{symbol}_{direction}.json"
    print(f"[START] {symbol} {direction}; profile={args.profile}; "
          f"population={evolution.smoke_population if args.profile == 'smoke' else evolution.main_population}; "
          f"generations={evolution.smoke_generations if args.profile == 'smoke' else evolution.main_generations}",
          flush=True)
    result = run_direction_evolution(
        symbol=symbol,
        direction=direction,
        evaluator=evaluator,
        checkpoint_path=checkpoint,
        checkpoint_metadata=metadata,
        evolution_config=evolution,
        profile=args.profile,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "metadata": metadata,
        "symbol": symbol,
        "direction": direction,
        "result": result,
        "oos_used_for_selection": False,
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[COMPLETE] {symbol} {direction}; eligible={result['eligible_phenotypes']} "
          f"evaluated={result['evaluated_phenotypes']}; elapsed={(time.monotonic()-started)/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
