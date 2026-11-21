"""Limited genetic evolution of execution filters/sizing on the Run #60 selected universe.

Signal-generation genes remain at the parity-tested baseline in this first stage.
The next evolution stage can expand them after these results are validated.
"""
from __future__ import annotations

import ast
import itertools
import json
import math
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

# The legacy SwingEngine is script-oriented; expose its modules explicitly
# while keeping this runner executable as a proper repository package.
SWING_ENGINE_DIR = str(Path(__file__).resolve().parents[1] / "SwingEngine")
if SWING_ENGINE_DIR not in sys.path:
    sys.path.insert(0, SWING_ENGINE_DIR)

from Backtest.GeneticEngine.config import EvolutionConfig, INITIAL_EQUITY
from Backtest.GeneticEngine.evaluator import evaluate_prepared_genome, prepare_genome_inputs
from Backtest.GeneticEngine.genome import StrategyGenome
from backtest_runner import ProgressMonitor, build_btc_trend_frames
from data_loader import TIMEFRAME_MS, load_okx_data
from okx_contract_specs import OKXContractSpecs

SELECTED_26 = (
    "ADA-USDT-SWAP", "APT-USDT-SWAP", "AVAX-USDT-SWAP", "BNB-USDT-SWAP",
    "BRETT-USDT-SWAP", "BTC-USDT-SWAP", "CHZ-USDT-SWAP", "COMP-USDT-SWAP",
    "DOGE-USDT-SWAP", "EGLD-USDT-SWAP", "ENJ-USDT-SWAP", "IOTA-USDT-SWAP",
    "JTO-USDT-SWAP", "JUP-USDT-SWAP", "MEME-USDT-SWAP", "ONDO-USDT-SWAP",
    "OP-USDT-SWAP", "POPCAT-USDT-SWAP", "SNX-USDT-SWAP", "SUSHI-USDT-SWAP",
    "THETA-USDT-SWAP", "TIA-USDT-SWAP", "TRX-USDT-SWAP", "XLM-USDT-SWAP",
    "XRP-USDT-SWAP", "ZIL-USDT-SWAP",
)
RISK_VALUES = (0.05, 0.25, 0.5)
RR_VALUES = (1.5, 2.0, 2.5)
BTC_FILTER_VALUES = (None, "1d_4h", "1d_1h", "1d_4h_1h")
POPULATION_SIZE = 12
GENERATIONS = 4
ELITES = 2
TOURNAMENT_SIZE = 3
TOP_OOS = 5


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _key(genome: StrategyGenome) -> str:
    return json.dumps(genome.to_dict(), sort_keys=True, separators=(",", ":"))


def _fitness(summary: dict) -> float:
    """Risk-aware training score; raw return is retained as a tie-break metric."""
    trades = int(summary.get("closed_trades", 0))
    if trades < 10:
        return -1_000.0 + trades
    pf = float(summary.get("profit_factor", 0.0))
    pf_term = math.log1p(min(pf, 10.0)) if math.isfinite(pf) else math.log1p(10.0)
    avg_r = float(summary.get("average_r", 0.0))
    win_rate = float(summary.get("win_rate_pct", 0.0))
    drawdown = float(summary.get("max_drawdown_pct", 100.0))
    return avg_r * 100.0 + pf_term * 10.0 + win_rate * 0.05 - drawdown * 0.5


def _random_genome(rng: random.Random, base: StrategyGenome) -> StrategyGenome:
    return replace(
        base,
        risk_percent=rng.choice(RISK_VALUES),
        reward_ratio=rng.choice(RR_VALUES),
        btc_filter_mode=rng.choice(BTC_FILTER_VALUES),
    )


def _mate(a: StrategyGenome, b: StrategyGenome, rng: random.Random) -> StrategyGenome:
    return replace(
        a,
        risk_percent=rng.choice((a.risk_percent, b.risk_percent)),
        reward_ratio=rng.choice((a.reward_ratio, b.reward_ratio)),
        btc_filter_mode=rng.choice((a.btc_filter_mode, b.btc_filter_mode)),
    )


def _mutate(genome: StrategyGenome, rng: random.Random) -> StrategyGenome:
    field = rng.choice(("risk_percent", "reward_ratio", "btc_filter_mode"))
    if field == "risk_percent":
        return replace(genome, risk_percent=rng.choice(RISK_VALUES))
    if field == "reward_ratio":
        return replace(genome, reward_ratio=rng.choice(RR_VALUES))
    return replace(genome, btc_filter_mode=rng.choice(BTC_FILTER_VALUES))


def _split_data(data: dict, split_ts: int):
    train_data = {}
    for symbol, item in data.items():
        frames = dict(item.frames)
        frame = item.frames["15m"]
        close_ts = frame["ts"].astype("int64") + TIMEFRAME_MS["15m"]
        frames["15m"] = frame.loc[close_ts <= split_ts].reset_index(drop=True)
        train_data[symbol] = SimpleNamespace(symbol=symbol, frames=frames)
    return train_data


def _split_signals(signals: dict, split_ts: int):
    train, test = {}, {}
    for symbol, frame in signals.items():
        if frame.empty:
            train[symbol] = frame.copy()
            test[symbol] = frame.copy()
        else:
            train[symbol] = frame.loc[frame["signal_ts"].astype("int64") <= split_ts].reset_index(drop=True)
            test[symbol] = frame.loc[frame["signal_ts"].astype("int64") > split_ts].reset_index(drop=True)
    return train, test


def _evaluate(genome, data, signals, obs, specs, btc_frames, initial_capital):
    return evaluate_prepared_genome(
        data, signals, obs, genome,
        contract_specs=specs,
        initial_capital=initial_capital,
        btc_trend_frames=btc_frames,
    )


def run(data_root="Backtest/data/okx", output_dir=None):
    out = Path(output_dir or os.environ.get(
        "GENETIC_OUTPUT_DIR", "Backtest/results/genetic_limited"
    ))
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "genetic_checkpoint.json"
    cache_path = out / "training_evaluation_cache.json"
    checkpoint = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checkpoint = {}
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}

    rng = random.Random(EvolutionConfig().random_seed)
    if checkpoint.get("random_state"):
        try:
            rng.setstate(ast.literal_eval(checkpoint["random_state"]))
        except (ValueError, SyntaxError, TypeError):
            pass
    monitor = ProgressMonitor(interval_seconds=30)
    monitor.start()
    try:
        monitor.set_stage("genetic_data_loading", "loading the validated 76-symbol OKX dataset")
        all_data = load_okx_data(data_root, expected_symbols=76)
        missing = sorted(set(SELECTED_26) - set(all_data))
        if missing:
            raise ValueError(f"Selected Run #60 symbols missing from dataset: {missing}")
        data = {symbol: all_data[symbol] for symbol in SELECTED_26}

        monitor.set_stage("genetic_contract_specs", "loading current OKX linear SWAP contract specifications")
        specs = OKXContractSpecs()
        specs.load()

        base = StrategyGenome.baseline()
        monitor.set_stage("genetic_signal_preparation", "preparing baseline signals with per-symbol resumable checkpoints")
        signals, obs = prepare_genome_inputs(
            data, base, progress_monitor=monitor,
            checkpoint_dir=out / "signal_symbol_checkpoints",
        )
        btc_frames = build_btc_trend_frames(data)

        btc_15m = data["BTC-USDT-SWAP"].frames["15m"]
        timestamps = (btc_15m["ts"].astype("int64") + TIMEFRAME_MS["15m"]).tolist()
        split_ts = int(timestamps[int(len(timestamps) * 0.70)])
        train_data = _split_data(data, split_ts)
        train_signals, test_signals = _split_signals(signals, split_ts)

        print(
            f"[GENETIC] limited search space=36 | population={POPULATION_SIZE} "
            f"| generations={GENERATIONS} | train cutoff={split_ts} | symbols={len(data)}",
            flush=True,
        )

        def evaluate(genome):
            key = _key(genome)
            if key in cache:
                return float(cache[key]["fitness"])
            summary, trades, skips = _evaluate(
                genome, train_data, train_signals, obs, specs, btc_frames, INITIAL_EQUITY
            )
            result = {
                "fitness": _fitness(summary),
                "summary": summary,
                "trade_rows": int(len(trades)),
                "skip_rows": int(len(skips)),
            }
            cache[key] = result
            _atomic_json(cache_path, cache)
            return float(result["fitness"])

        population = []
        saved_population = checkpoint.get("population")
        if saved_population:
            try:
                population = [StrategyGenome.from_dict(x) for x in saved_population]
            except Exception:
                population = []
        if not population:
            population = [_random_genome(rng, base) for _ in range(POPULATION_SIZE)]
        history = checkpoint.get("history", [])
        start_generation = int(checkpoint.get("next_generation", 0))

        for generation in range(start_generation, GENERATIONS):
            monitor.set_stage("genetic_evolution", f"generation {generation + 1}/{GENERATIONS}")
            scored = [(evaluate(genome), genome) for genome in population]
            scored.sort(key=lambda item: item[0], reverse=True)
            history.append({
                "generation": generation + 1,
                "best_fitness": scored[0][0],
                "best_genome": scored[0][1].to_dict(),
                "unique_evaluations": len(cache),
            })
            print(
                f"[GENERATION] {generation + 1}/{GENERATIONS} best_fitness={scored[0][0]:.5f} "
                f"genome={scored[0][1].to_dict()} unique={len(cache)}",
                flush=True,
            )
            next_population = [item[1] for item in scored[:ELITES]]
            while len(next_population) < POPULATION_SIZE:
                parent_a = max(rng.sample(scored, min(TOURNAMENT_SIZE, len(scored))), key=lambda x: x[0])[1]
                parent_b = max(rng.sample(scored, min(TOURNAMENT_SIZE, len(scored))), key=lambda x: x[0])[1]
                child = _mate(parent_a, parent_b, rng)
                if rng.random() < 0.35:
                    child = _mutate(child, rng)
                next_population.append(child)
            population = next_population
            _atomic_json(checkpoint_path, {
                "next_generation": generation + 1,
                "population": [item.to_dict() for item in population],
                "history": history,
                "split_ts": split_ts,
                "completed_generations": generation + 1,
                "random_state": repr(rng.getstate()),
            })

        candidates = sorted(
            ((float(value["fitness"]), StrategyGenome.from_dict(json.loads(key)))
             for key, value in cache.items()),
            key=lambda item: item[0],
            reverse=True,
        )
        top_genomes = []
        seen = set()
        for _, genome in candidates:
            key = _key(genome)
            if key not in seen:
                top_genomes.append(genome)
                seen.add(key)
            if len(top_genomes) >= TOP_OOS:
                break
        if _key(base) not in seen:
            top_genomes.append(base)

        monitor.set_stage("genetic_oos_validation", "validating top evolved candidates on untouched chronological holdout")
        validation_rows = []
        for index, genome in enumerate(top_genomes, 1):
            summary, trades, skips = _evaluate(
                genome, data, test_signals, obs, specs, btc_frames, INITIAL_EQUITY
            )
            validation_rows.append({
                "rank": index,
                "genome": _key(genome),
                "fitness_train": cache.get(_key(genome), {}).get("fitness", None),
                **summary,
                "oos_trade_rows": int(len(trades)),
                "oos_skip_rows": int(len(skips)),
            })
            trades.to_csv(out / f"oos_trades_rank_{index}.csv", index=False)
            print(
                f"[OOS] rank={index} final_equity={summary.get('final_equity')} "
                f"trades={len(trades)} genome={_key(genome)}",
                flush=True,
            )

        pd.DataFrame([
            {"genome": key, **value["summary"], "fitness": value["fitness"],
             "training_trade_rows": value["trade_rows"]}
            for key, value in cache.items()
        ]).sort_values("fitness", ascending=False).to_csv(
            out / "genetic_limited_training_scores.csv", index=False
        )
        pd.DataFrame(validation_rows).to_csv(
            out / "genetic_limited_oos_validation.csv", index=False
        )
        (out / "README.md").write_text(
            "# Limited Genetic Evolution — OOS validation\n\n"
            "- Universe: Run #60 selected 26 symbols\n- Initial equity: USD 100.00\n"
            f"- Chronological training split: 70%; cutoff timestamp: {split_ts}\n"
            f"- Genetic population: {POPULATION_SIZE}; generations: {GENERATIONS}; seed: {EvolutionConfig().random_seed}\n"
            "- Evolved genes in this bounded stage: BTC filter mode, risk percentage and reward/risk ratio.\n"
            "- Signal-generation genes remain at parity-tested baseline; no unsupported genes are silently mutated.\n"
            "- OOS holdout is untouched during evolution; top candidates are evaluated only after evolution.\n"
            "- Fees and slippage are zero.\n"
            "- Checkpoints preserve per-symbol signal preparation, candidate fitness cache and completed generations.\n",
            encoding="utf-8",
        )
        checkpoint["complete"] = True
        checkpoint["completed_generations"] = GENERATIONS
        _atomic_json(checkpoint_path, checkpoint)
        print(f"[GENETIC COMPLETE] unique training candidates={len(cache)}; OOS candidates={len(validation_rows)}", flush=True)
    finally:
        monitor.stop()


if __name__ == "__main__":
    run()
