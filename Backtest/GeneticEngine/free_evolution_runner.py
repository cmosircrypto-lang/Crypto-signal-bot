"""Free genetic evolution over signal-structure genes supported by SwingEngine.

Only genes actually wired through prepare_signals are evolvable here. All
other genome fields stay at legacy defaults. Training is chronological;
the final 30% holdout is evaluated only after evolution and never feeds fitness.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SWING_ENGINE_DIR = str(Path(__file__).resolve().parents[1] / "SwingEngine")
if SWING_ENGINE_DIR not in sys.path:
    sys.path.insert(0, SWING_ENGINE_DIR)

from Backtest.GeneticEngine.config import INITIAL_EQUITY, EvolutionConfig
from Backtest.GeneticEngine.evaluator import prepare_genome_inputs, evaluate_prepared_genome
from Backtest.GeneticEngine.genome import StrategyGenome
from Backtest.SwingEngine.backtest_runner import ProgressMonitor, build_btc_trend_frames
from Backtest.SwingEngine.data_loader import TIMEFRAME_MS, load_okx_data
from Backtest.SwingEngine.okx_contract_specs import OKXContractSpecs

SELECTED_26 = (
    "ADA-USDT-SWAP", "APT-USDT-SWAP", "AVAX-USDT-SWAP", "BNB-USDT-SWAP",
    "BRETT-USDT-SWAP", "BTC-USDT-SWAP", "CHZ-USDT-SWAP", "COMP-USDT-SWAP",
    "DOGE-USDT-SWAP", "EGLD-USDT-SWAP", "ENJ-USDT-SWAP", "IOTA-USDT-SWAP",
    "JTO-USDT-SWAP", "JUP-USDT-SWAP", "MEME-USDT-SWAP", "ONDO-USDT-SWAP",
    "OP-USDT-SWAP", "POPCAT-USDT-SWAP", "SNX-USDT-SWAP", "SUSHI-USDT-SWAP",
    "THETA-USDT-SWAP", "TIA-USDT-SWAP", "TRX-USDT-SWAP", "XLM-USDT-SWAP",
    "XRP-USDT-SWAP", "ZIL-USDT-SWAP",
)
METHODS = ("DIRECT", "ENGULFING", "PIN_BAR", "M15_STRUCTURE_BREAK")
METHOD_SETS = tuple(
    tuple(combo)
    for size in range(1, len(METHODS) + 1)
    for combo in itertools.combinations(METHODS, size)
)
ZONES = ("body", "wick")
TOUCHES = ("first", "max_two", "progressive")
ALIGNMENTS = (False, True)
RISKS = (0.05, 0.1, 0.25, 0.5)
RRS = (1.5, 2.0, 2.5, 3.0)
FILTERS = (None, "1d_4h", "1d_1h", "1d_4h_1h")
POPULATION = 12
GENERATIONS = 5
ELITES = 2
TOURNAMENT = 3
TOP_OOS = 3


def _key(genome):
    return json.dumps(genome.to_dict(), sort_keys=True, separators=(",", ":"))


def _fitness(summary):
    trades = int(summary.get("closed_trades", 0))
    if trades < 20:
        return -1000.0 + trades
    pf = float(summary.get("profit_factor", 0.0))
    avg_r = float(summary.get("average_r", 0.0))
    win = float(summary.get("win_rate_pct", 0.0))
    dd = float(summary.get("max_drawdown_pct", 100.0))
    if not pd.notna(pf):
        pf = 0.0
    return avg_r * 100.0 + min(pf, 5.0) * 8.0 + win * 0.03 - dd * 0.75


def _signal_key(g):
    payload = {
        "entry_methods": list(g.entry_methods), "zone_type": g.zone_type,
        "touch_rule": g.touch_rule, "require_4h_alignment": g.require_4h_alignment,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _random_genome(rng):
    return StrategyGenome(
        entry_methods=rng.choice(METHOD_SETS),
        zone_type=rng.choice(ZONES),
        touch_rule=rng.choice(TOUCHES),
        require_4h_alignment=rng.choice(ALIGNMENTS),
        risk_percent=rng.choice(RISKS),
        reward_ratio=rng.choice(RRS),
        btc_filter_mode=rng.choice(FILTERS),
    )


def _mate(a, b, rng):
    return StrategyGenome(
        entry_methods=rng.choice((a.entry_methods, b.entry_methods)),
        zone_type=rng.choice((a.zone_type, b.zone_type)),
        touch_rule=rng.choice((a.touch_rule, b.touch_rule)),
        require_4h_alignment=rng.choice((a.require_4h_alignment, b.require_4h_alignment)),
        risk_percent=rng.choice((a.risk_percent, b.risk_percent)),
        reward_ratio=rng.choice((a.reward_ratio, b.reward_ratio)),
        btc_filter_mode=rng.choice((a.btc_filter_mode, b.btc_filter_mode)),
    )


def _mutate(g, rng):
    field = rng.choice(("entry_methods", "zone_type", "touch_rule",
                        "require_4h_alignment", "risk_percent",
                        "reward_ratio", "btc_filter_mode"))
    choices = {
        "entry_methods": METHODS_SETS_SAFE,
        "zone_type": ZONES, "touch_rule": TOUCHES,
        "require_4h_alignment": ALIGNMENTS, "risk_percent": RISKS,
        "reward_ratio": RRS, "btc_filter_mode": FILTERS,
    }
    return replace(g, **{field: rng.choice(choices[field])})


METHODS_SETS_SAFE = METHOD_SETS


def _split_data(data, split_ts):
    result = {}
    for symbol, item in data.items():
        frames = dict(item.frames)
        frame = item.frames["15m"]
        close_ts = frame["ts"].astype("int64") + TIMEFRAME_MS["15m"]
        frames["15m"] = frame.loc[close_ts <= split_ts].reset_index(drop=True)
        result[symbol] = SimpleNamespace(symbol=symbol, frames=frames)
    return result


def _split_signals(signals, split_ts):
    train, holdout = {}, {}
    for symbol, frame in signals.items():
        if frame.empty:
            train[symbol], holdout[symbol] = frame.copy(), frame.copy()
        else:
            train[symbol] = frame.loc[
                frame["signal_ts"].astype("int64") <= split_ts
            ].reset_index(drop=True)
            holdout[symbol] = frame.loc[
                frame["signal_ts"].astype("int64") > split_ts
            ].reset_index(drop=True)
    return train, holdout


def _atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run(data_root="Backtest/data/okx", output_dir=None):
    out = Path(output_dir or os.environ.get(
        "GENETIC_OUTPUT_DIR", "Backtest/results/genetic_free"
    ))
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "free_evolution_checkpoint.json"
    cache_path = out / "free_training_cache.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rng = random.Random(EvolutionConfig().random_seed + 1)
    if checkpoint.get("random_state"):
        try:
            rng.setstate(ast.literal_eval(checkpoint["random_state"]))
        except (ValueError, SyntaxError, TypeError):
            pass

    monitor = ProgressMonitor(interval_seconds=30)
    monitor.start()
    signal_cache = {}
    try:
        monitor.set_stage("free_data_loading", "loading cached 76-symbol OKX dataset")
        all_data = load_okx_data(data_root, expected_symbols=76)
        missing = sorted(set(SELECTED_26) - set(all_data))
        if missing:
            raise ValueError(f"Missing selected symbols: {missing}")
        data = {s: all_data[s] for s in SELECTED_26}
        specs = OKXContractSpecs()
        specs.load()
        btc_frames = build_btc_trend_frames(data)
        btc = data["BTC-USDT-SWAP"].frames["15m"]
        closes = (btc["ts"].astype("int64") + TIMEFRAME_MS["15m"]).tolist()
        split_ts = int(closes[int(len(closes) * 0.70)])
        train_data = _split_data(data, split_ts)

        def prepared(g):
            skey = _signal_key(g)
            if skey not in signal_cache:
                monitor.set_stage("free_signal_preparation", f"preparing signal phenotype {skey}")
                signals, obs = prepare_genome_inputs(
                    data, g, progress_monitor=monitor,
                    checkpoint_dir=out / "signal_cache" / skey,
                )
                train_signals, holdout_signals = _split_signals(signals, split_ts)
                signal_cache[skey] = (signals, obs, train_signals, holdout_signals)
            return signal_cache[skey]

        def evaluate(g):
            key = _key(g)
            if key in cache:
                return float(cache[key]["fitness"])
            _, obs, train_signals, _ = prepared(g)
            summary, trades, _system = evaluate_prepared_genome(
                train_data, train_signals, obs, g, contract_specs=specs,
                initial_capital=INITIAL_EQUITY, btc_trend_frames=btc_frames,
            )
            score = _fitness(summary)
            cache[key] = {"fitness": score, "summary": summary, "training_trade_rows": len(trades)}
            _atomic_json(cache_path, cache)
            print(f"[FREE CANDIDATE] fitness={score:.5f} pnl={summary.get('net_pnl')} "
                  f"trades={len(trades)} signal={_signal_key(g)}", flush=True)
            return score

        population = [StrategyGenome.from_dict(x) for x in checkpoint.get("population", [])]
        if not population:
            population = [_random_genome(rng) for _ in range(POPULATION)]
        history = checkpoint.get("history", [])
        start = int(checkpoint.get("next_generation", 0))
        for gen in range(start, GENERATIONS):
            monitor.set_stage("free_genetic_evolution", f"generation {gen + 1}/{GENERATIONS}")
            scored = [(evaluate(g), g) for g in population]
            scored.sort(key=lambda x: x[0], reverse=True)
            history.append({
                "generation": gen + 1, "best_fitness": scored[0][0],
                "best_genome": scored[0][1].to_dict(),
                "unique_candidates": len(cache), "signal_phenotypes": len(signal_cache),
            })
            print(f"[FREE GENERATION] {gen + 1}/{GENERATIONS} best={scored[0][0]:.5f} "
                  f"genome={_key(scored[0][1])}", flush=True)
            nxt = [g for _, g in scored[:ELITES]]
            while len(nxt) < POPULATION:
                pa = max(rng.sample(scored, min(TOURNAMENT, len(scored))), key=lambda x: x[0])[1]
                pb = max(rng.sample(scored, min(TOURNAMENT, len(scored))), key=lambda x: x[0])[1]
                child = _mate(pa, pb, rng)
                if rng.random() < 0.35:
                    child = _mutate(child, rng)
                child.validate()
                nxt.append(child)
            population = nxt
            _atomic_json(checkpoint_path, {
                "next_generation": gen + 1, "population": [g.to_dict() for g in population],
                "history": history, "split_ts": split_ts,
                "random_state": repr(rng.getstate()),
            })

        candidates = sorted(
            ((float(v["fitness"]), StrategyGenome.from_dict(json.loads(k)))
             for k, v in cache.items()), key=lambda x: x[0], reverse=True,
        )
        top = []
        seen = set()
        for _, g in candidates:
            if _key(g) not in seen:
                top.append(g); seen.add(_key(g))
            if len(top) >= TOP_OOS:
                break

        rows = []
        for rank, g in enumerate(top, 1):
            _, obs, _, holdout_signals = prepared(g)
            summary, trades, _system = evaluate_prepared_genome(
                data, holdout_signals, obs, g, contract_specs=specs,
                initial_capital=INITIAL_EQUITY, btc_trend_frames=btc_frames,
            )
            rows.append({"rank": rank, "genome": _key(g),
                         "fitness_train": cache[_key(g)]["fitness"], **summary,
                         "oos_trade_rows": len(trades)})
            trades.to_csv(out / f"free_oos_trades_rank_{rank}.csv", index=False)
            print(f"[FREE OOS] rank={rank} equity={summary.get('final_equity')} "
                  f"trades={len(trades)} genome={_key(g)}", flush=True)

        pd.DataFrame(rows).to_csv(out / "genetic_free_oos_validation.csv", index=False)
        pd.DataFrame([
            {"genome": k, **v["summary"], "fitness": v["fitness"],
             "training_trade_rows": v["training_trade_rows"]}
            for k, v in cache.items()
        ]).sort_values("fitness", ascending=False).to_csv(
            out / "genetic_free_training_scores.csv", index=False
        )
        (out / "README.md").write_text(
            "# Free Genetic Evolution\n\n"
            "- Evolvable genes: entry-method subset, zone type, touch rule, 4h alignment, "
            "risk percentage, reward/risk, BTC filter.\n"
            "- Swing/OB/ATR/stop internals remain fixed at legacy-compatible defaults.\n"
            "- Universe: same 26 Run #60 symbols; initial equity $100.\n"
            f"- Chronological training split: 70%; cutoff: {split_ts}.\n"
            f"- Population: {POPULATION}; generations: {GENERATIONS}; seed: "
            f"{EvolutionConfig().random_seed + 1}.\n"
            "- OOS holdout is excluded from fitness and evaluated only after evolution.\n"
            "- Fees, slippage and funding remain zero; simulation only, no orders.\n"
            "- Signal logic in the live bot is untouched.\n",
            encoding="utf-8",
        )
        checkpoint.update({"complete": True, "completed_generations": GENERATIONS})
        _atomic_json(checkpoint_path, checkpoint)
        print(f"[FREE GENETIC COMPLETE] candidates={len(cache)} signal_phenotypes={len(signal_cache)} "
              f"oos={len(rows)}", flush=True)
    finally:
        monitor.stop()


if __name__ == "__main__":
    run()
