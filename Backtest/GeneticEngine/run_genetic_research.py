"""CLI for resumable walk-forward genetic research; simulation only, no orders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

from . import config
from .genetic_runner import run_genetic_evolution


def _source_hash(root: Path) -> str:
    paths = [
        root/"Backtest/GeneticEngine/config.py",
        root/"Backtest/GeneticEngine/genome.py",
        root/"Backtest/GeneticEngine/evolution.py",
        root/"Backtest/GeneticEngine/family_signals.py",
        root/"Backtest/GeneticEngine/historical_evaluator.py",
        root/"Backtest/GeneticEngine/genetic_runner.py",
        root/"Backtest/GeneticEngine/checkpoint.py",
    ]
    digest=hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_hash(data) -> str:
    import pandas as pd
    digest=hashlib.sha256()
    for symbol,item in sorted(data.items()):
        digest.update(symbol.encode())
        for timeframe,frame in sorted(item.frames.items()):
            digest.update(timeframe.encode())
            digest.update(pd.util.hash_pandas_object(frame,index=False).values.tobytes())
    return digest.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",default="Backtest/data/okx")
    parser.add_argument("--checkpoint-dir",default="Backtest/results/genetic_family_checkpoints")
    parser.add_argument("--output",default="Backtest/results/genetic_family_summary.json")
    parser.add_argument("--symbols",default="",help="Comma-separated symbols; empty means all loaded symbols")
    parser.add_argument("--profile",choices=("smoke","main"),default="smoke")
    parser.add_argument("--population",type=int,default=8,help="Population size for smoke profile")
    parser.add_argument("--generations",type=int,default=2,help="Generations for smoke profile")
    parser.add_argument("--seed",type=int,default=20261121)
    parser.add_argument("--expected-symbols",type=int,default=76)
    parser.add_argument("--fold-count",type=int,default=3)
    parser.add_argument("--oos-fraction",type=float,default=0.15)
    parser.add_argument("--warmup-fraction",type=float,default=0.50)
    args=parser.parse_args(argv)
    if args.population<2 or args.generations<1:
        parser.error("population must be >=2 and generations >=1")

    root=Path.cwd()
    swing_path=root/"Backtest/SwingEngine"
    sys.path.insert(0,str(swing_path))
    from data_loader import load_okx_data

    started=time.monotonic()
    print(f"[LOAD] validating cached dataset at {args.data_root}",flush=True)
    data=load_okx_data(args.data_root,expected_symbols=args.expected_symbols)
    symbols=sorted(data)
    if args.symbols:
        requested={x.strip().upper() for x in args.symbols.split(",") if x.strip()}
        missing=requested-set(data)
        if missing:
            parser.error(f"unknown symbols: {sorted(missing)}")
        symbols=sorted(requested)
    if args.profile=="smoke":
        evolution=config.EvolutionConfig(
            random_seed=args.seed,smoke_population=args.population,
            smoke_generations=args.generations,tournament_size=2,
            crossover_probability=0.8,mutation_probability=0.2,
        )
    else:
        evolution=config.EvolutionConfig(random_seed=args.seed)
    evolution.validate()

    print(f"[HASH] hashing {len(data)} symbols and research source",flush=True)
    dataset_hash=_dataset_hash(data)
    code_hash=_source_hash(root)
    config_payload={
        "evolution":asdict(evolution),"profile":args.profile,
        "fold_count":args.fold_count,"oos_fraction":args.oos_fraction,
        "warmup_fraction":args.warmup_fraction,
        "fees":0.0,"slippage":0.0,"same_candle_policy":"SL_FIRST",
    }
    config_hash=hashlib.sha256(json.dumps(config_payload,sort_keys=True).encode()).hexdigest()
    metadata={"code_hash":code_hash,"config_hash":config_hash,"dataset_hash":dataset_hash}
    print(f"[START] symbols={len(symbols)} directions=2 profile={args.profile} "
          f"population={evolution.smoke_population if args.profile=='smoke' else evolution.main_population} "
          f"generations={evolution.smoke_generations if args.profile=='smoke' else evolution.main_generations}",flush=True)
    results=run_genetic_evolution(
        all_symbol_data=data,symbols=symbols,
        checkpoint_dir=args.checkpoint_dir,checkpoint_metadata=metadata,
        evolution_config=evolution,profile=args.profile,
        oos_fraction=args.oos_fraction,warmup_fraction=args.warmup_fraction,
        fold_count=args.fold_count,
    )
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({
        "metadata":metadata,"config":config_payload,"results":results,
        "oos_used_for_selection":False,"elapsed_seconds":time.monotonic()-started,
    },indent=2,sort_keys=True),encoding="utf-8")
    print(f"[COMPLETE] wrote {output}; elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)
    for symbol in sorted(results):
        for direction in ("LONG","SHORT"):
            item=results[symbol][direction]
            champ=item["champion"]
            print(f"[RESULT] {symbol} {direction}: eligible={item['eligible_phenotypes']} "
                  f"evaluated={item['evaluated_phenotypes']} champion={'yes' if champ else 'NO'}",flush=True)


if __name__=="__main__":
    main()
