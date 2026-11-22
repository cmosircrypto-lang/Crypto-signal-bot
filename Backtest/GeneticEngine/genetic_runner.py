"""Resumable genetic population evolution, independent per symbol and direction."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping

from . import config
from .checkpoint import load_checkpoint, save_checkpoint
from .dynamic_selector import HistoricalMetrics, historical_quality_score
from .evolution import crossover_genomes, mutate_genome, random_genome
from .genome import StrategyGenome


_METRIC_FIELDS = (
    "average_r", "max_drawdown_pct", "profit_factor", "positive_window_ratio",
    "trade_count", "validation_score", "complexity_penalty",
)


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _genome_key(genome: StrategyGenome) -> str:
    return _canonical_hash(genome.to_dict())


def _metrics_only(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(_METRIC_FIELDS) - set(value)
    if missing:
        raise ValueError(f"historical evaluator omitted metrics: {sorted(missing)}")
    metrics = {name: value[name] for name in _METRIC_FIELDS}
    HistoricalMetrics(**metrics).validate()
    return metrics


def _fitness(metrics: Mapping[str, Any], *, min_trades: int, max_drawdown_pct: float,
             min_profit_factor: float, min_positive_window_ratio: float,
             min_validation_score: float) -> tuple[bool, float]:
    parsed = HistoricalMetrics(**dict(metrics))
    score = historical_quality_score(parsed)
    eligible = (
        parsed.trade_count >= min_trades
        and parsed.max_drawdown_pct <= max_drawdown_pct
        and parsed.profit_factor >= min_profit_factor
        and parsed.positive_window_ratio >= min_positive_window_ratio
        and parsed.validation_score >= min_validation_score
    )
    if parsed.trade_count == 0:
        return False, -1.0
    penalty = 0.0
    if parsed.trade_count < min_trades:
        penalty += 0.25 * (1.0 - parsed.trade_count / min_trades)
    if parsed.max_drawdown_pct > max_drawdown_pct:
        penalty += min(0.35, (parsed.max_drawdown_pct-max_drawdown_pct)/100.0)
    if parsed.profit_factor < min_profit_factor:
        penalty += min(0.25, (min_profit_factor-parsed.profit_factor)*0.2)
    if parsed.positive_window_ratio < min_positive_window_ratio:
        penalty += 0.15 * (min_positive_window_ratio-parsed.positive_window_ratio)
    if parsed.validation_score < min_validation_score:
        penalty += 0.15 * (min_validation_score-parsed.validation_score)
    return eligible, score-penalty


def _rank_key(item: Mapping[str, Any]):
    return (bool(item["eligible"]), float(item["fitness"]), int(item["metrics"]["trade_count"]),
            _genome_key(StrategyGenome.from_dict(item["genome"])))


def _seed_population(symbol: str, size: int, seed: int,
                     seed_genomes: Iterable[StrategyGenome] = ()) -> list[StrategyGenome]:
    rng = random.Random(seed)
    population: list[StrategyGenome] = []
    seen: set[str] = set()

    def add(genome: StrategyGenome):
        candidate = replace(genome, symbols=(symbol,))
        candidate.validate()
        key = _genome_key(candidate)
        if key not in seen and len(population) < size:
            seen.add(key)
            population.append(candidate)

    for genome in seed_genomes:
        add(genome)
    baseline = StrategyGenome.baseline()
    for family in config.STRATEGY_FAMILIES:
        add(replace(baseline, signal_families=(family,), family_combination="SINGLE"))
    for i, left in enumerate(config.STRATEGY_FAMILIES):
        for right in config.STRATEGY_FAMILIES[i+1:]:
            add(replace(baseline, signal_families=(left, right),
                        family_combination="VOTE", min_family_agreement=2))
    attempts = 0
    while len(population) < size and attempts < size*50:
        attempts += 1
        add(random_genome(rng, symbols=(symbol,)))
    if len(population) < size:
        raise RuntimeError("could not create a unique initial genetic population")
    return population


def _tournament(records: list[dict[str, Any]], rng: random.Random, size: int) -> StrategyGenome:
    sample = rng.sample(records, min(size, len(records)))
    winner = max(sample, key=_rank_key)
    return StrategyGenome.from_dict(winner["genome"])


def run_direction_evolution(
    *, symbol: str, direction: str,
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    checkpoint_path: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    evolution_config: config.EvolutionConfig = config.EvolutionConfig(),
    profile: str = "smoke",
    seed_genomes: Iterable[StrategyGenome] = (),
    elite_fraction: float = 0.10,
    min_trades: int = 30,
    max_drawdown_pct: float = 20.0,
    min_profit_factor: float = 1.0,
    min_positive_window_ratio: float = 0.5,
    min_validation_score: float = 0.45,
) -> dict[str, Any]:
    """Evolve one symbol and one direction with atomic, resumable checkpoints."""
    symbol, direction = symbol.strip().upper(), direction.strip().upper()
    if not symbol or direction not in ("LONG", "SHORT"):
        raise ValueError("valid symbol and LONG/SHORT direction are required")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    evolution_config.validate()
    if profile not in ("smoke", "main"):
        raise ValueError("profile must be smoke or main")
    if not 0.0 < elite_fraction < 1.0:
        raise ValueError("elite_fraction must be between zero and one")
    population_size = evolution_config.smoke_population if profile == "smoke" else evolution_config.main_population
    generations = evolution_config.smoke_generations if profile == "smoke" else evolution_config.main_generations
    required_meta = {"code_hash", "config_hash", "dataset_hash"}
    if required_meta-set(checkpoint_metadata):
        raise ValueError(f"checkpoint metadata missing: {sorted(required_meta-set(checkpoint_metadata))}")

    seed_list = tuple(seed_genomes)
    metadata = {
        **dict(checkpoint_metadata), "symbol": symbol, "direction": direction,
        "random_seed": evolution_config.random_seed, "profile": profile,
        "population_size": population_size, "generations": generations,
        "elite_fraction": elite_fraction, "min_trades": min_trades,
        "evolution_config": asdict(evolution_config),
        "seed_genomes_hash": _canonical_hash([g.to_dict() for g in seed_list]),
        "fitness_gates": {
            "max_drawdown_pct": max_drawdown_pct,
            "min_profit_factor": min_profit_factor,
            "min_positive_window_ratio": min_positive_window_ratio,
            "min_validation_score": min_validation_score,
        },
    }
    path = Path(checkpoint_path)
    rng_seed = evolution_config.random_seed + sum((i+1)*ord(ch) for i,ch in enumerate(symbol))
    rng_seed += 0 if direction == "LONG" else 1_000_003

    if path.exists():
        restored = load_checkpoint(path, expected_metadata=metadata)
        payload = restored["payload"]
        if payload.get("status") == "COMPLETED":
            return payload["result"]
        generation = int(payload["generation"])
        population = [StrategyGenome.from_dict(x) for x in payload["population"]]
        evaluated = dict(payload.get("evaluated", {}))
        history = list(payload.get("history", []))
    else:
        generation = 0
        population = _seed_population(symbol, population_size, rng_seed, seed_list)
        evaluated: dict[str, dict[str, Any]] = {}
        history: list[dict[str, Any]] = []
        save_checkpoint(path, metadata=metadata, payload={
            "status": "RUNNING", "generation": generation,
            "population": [g.to_dict() for g in population],
            "evaluated": evaluated, "history": history,
        })

    for gen in range(generation, generations):
        generation_records = []
        for index, genome in enumerate(population):
            key = _genome_key(genome)
            if key not in evaluated:
                spec = {
                    "candidate_id": key[:20], "symbol": symbol, "direction": direction,
                    "genome": genome.to_dict(),
                }
                raw_metrics = evaluator(spec)
                if not isinstance(raw_metrics, Mapping):
                    raise TypeError("historical evaluator must return a metrics mapping")
                metrics = _metrics_only(raw_metrics)
                eligible, fitness = _fitness(
                    metrics, min_trades=min_trades, max_drawdown_pct=max_drawdown_pct,
                    min_profit_factor=min_profit_factor,
                    min_positive_window_ratio=min_positive_window_ratio,
                    min_validation_score=min_validation_score,
                )
                evaluated[key] = {
                    "genome": genome.to_dict(), "metrics": metrics,
                    "eligible": eligible, "fitness": fitness,
                }
                save_checkpoint(path, metadata=metadata, payload={
                    "status": "RUNNING", "generation": gen,
                    "population": [g.to_dict() for g in population],
                    "evaluated": evaluated, "history": history,
                    "next_candidate_index": index+1,
                })
            generation_records.append(evaluated[key])
            if (index+1) % 5 == 0 or index+1 == len(population):
                print(f"[GENETIC] {symbol} {direction} generation {gen+1}/{generations}: "
                      f"evaluated {index+1}/{len(population)}", flush=True)

        ranked = sorted(generation_records, key=_rank_key, reverse=True)
        best = ranked[0]
        history.append({
            "generation": gen, "best_fitness": float(best["fitness"]),
            "best_eligible": bool(best["eligible"]),
            "eligible_candidates": sum(bool(x["eligible"]) for x in generation_records),
            "best_trade_count": int(best["metrics"]["trade_count"]),
            "best_profit_factor": float(best["metrics"]["profit_factor"]),
            "best_drawdown_pct": float(best["metrics"]["max_drawdown_pct"]),
        })
        print(f"[GENETIC] {symbol} {direction} generation {gen+1}/{generations} complete: "
              f"eligible={history[-1]['eligible_candidates']}/{population_size}, "
              f"best_fitness={history[-1]['best_fitness']:.4f}", flush=True)
        if gen+1 >= generations:
            break

        rng = random.Random(rng_seed + (gen+1)*7919)
        elite_count = max(1, min(population_size-1, int(population_size*elite_fraction)))
        next_population = [
            StrategyGenome.from_dict(item["genome"]) for item in ranked[:elite_count]
        ]
        seen = {_genome_key(x) for x in next_population}
        attempts = 0
        while len(next_population) < population_size and attempts < population_size*100:
            attempts += 1
            parent_a = _tournament(ranked, rng, evolution_config.tournament_size)
            if rng.random() < evolution_config.crossover_probability:
                parent_b = _tournament(ranked, rng, evolution_config.tournament_size)
                child = crossover_genomes(parent_a, parent_b, rng)
            else:
                child = parent_a
            child = mutate_genome(child, rng, evolution_config.mutation_probability)
            child = replace(child, symbols=(symbol,))
            key = _genome_key(child)
            if key in seen:
                continue
            seen.add(key)
            next_population.append(child)
        while len(next_population) < population_size:
            child = random_genome(rng, symbols=(symbol,))
            key = _genome_key(child)
            if key not in seen:
                seen.add(key)
                next_population.append(child)

        population = next_population
        save_checkpoint(path, metadata=metadata, payload={
            "status": "RUNNING", "generation": gen+1,
            "population": [g.to_dict() for g in population],
            "evaluated": evaluated, "history": history,
        })

    all_records = sorted(evaluated.values(), key=_rank_key, reverse=True)
    eligible_records = [x for x in all_records if x["eligible"]]
    result = {
        "symbol": symbol, "direction": direction, "generations": generations,
        "population_size": population_size,
        "evaluated_phenotypes": len(evaluated),
        "eligible_phenotypes": len(eligible_records),
        "champion": eligible_records[0] if eligible_records else None,
        "best_exploratory": all_records[0] if all_records else None,
        "history": history,
        "oos_used_for_selection": False,
    }
    save_checkpoint(path, metadata=metadata, payload={
        "status": "COMPLETED", "generation": generations,
        "population": [g.to_dict() for g in population],
        "evaluated": evaluated, "history": history, "result": result,
    })
    return result


def run_genetic_evolution(
    *, all_symbol_data: Mapping[str, Any], symbols: Iterable[str],
    checkpoint_dir: str | Path, checkpoint_metadata: Mapping[str, Any],
    evolution_config: config.EvolutionConfig = config.EvolutionConfig(),
    profile: str = "smoke", seed_genomes: Iterable[StrategyGenome] = (),
    oos_fraction: float = 0.15, warmup_fraction: float = 0.50,
    fold_count: int = 3, swing_signal_provider=None,
) -> dict[str, dict[str, Any]]:
    """Run independent LONG and SHORT populations for each requested symbol."""
    from .historical_evaluator import evaluate_candidate_walk_forward

    root = Path(checkpoint_dir)
    seed_list = tuple(seed_genomes)
    run_metadata = {
        **dict(checkpoint_metadata),
        "walk_forward": {
            "oos_fraction": oos_fraction, "warmup_fraction": warmup_fraction,
            "fold_count": fold_count,
        },
        "swing_signal_provider": (
            getattr(swing_signal_provider, "__qualname__", "custom")
            if callable(swing_signal_provider) else None
        ),
    }
    results = {}
    for symbol in sorted({s.strip().upper() for s in symbols}):
        if symbol not in all_symbol_data:
            raise KeyError(f"symbol not found in dataset: {symbol}")

        def evaluator(spec):
            genome = StrategyGenome.from_dict(spec["genome"])
            return evaluate_candidate_walk_forward(
                all_symbol_data, symbol, genome, spec["direction"],
                oos_fraction=oos_fraction, warmup_fraction=warmup_fraction,
                fold_count=fold_count, swing_signal_provider=swing_signal_provider,
            )

        results[symbol] = {}
        for direction in ("LONG", "SHORT"):
            results[symbol][direction] = run_direction_evolution(
                symbol=symbol, direction=direction, evaluator=evaluator,
                checkpoint_path=root/f"{symbol}_{direction}.json",
                checkpoint_metadata=run_metadata,
                evolution_config=evolution_config, profile=profile,
                seed_genomes=seed_list,
            )
    return results
