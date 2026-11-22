"""Resumable per-symbol evaluation of composable, direction-aware candidates."""
from __future__ import annotations
import hashlib
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from . import config
from .checkpoint import load_checkpoint, save_checkpoint
from .genome import StrategyGenome
from .dynamic_selector import HistoricalMetrics, ScoringPolicy, SignalCandidate, SelectionDecision, select_cycle_signals

DIRECTIONS = ("LONG", "SHORT")

def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _candidate_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("candidate_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("each candidate requires a non-empty candidate_id")
    return value

def _to_signal_candidate(symbol: str, spec: Mapping[str, Any], result: Mapping[str, Any]) -> SignalCandidate:
    historical = result.get("historical")
    if not isinstance(historical, Mapping):
        raise ValueError("evaluator result must include historical metrics mapping")
    expected = spec.get("direction")
    actual = str(result.get("direction", expected or "LONG")).upper()
    if expected is not None and actual != str(expected).upper():
        raise ValueError(f"direction-specific evaluator mismatch: expected {expected}, got {actual}")
    return SignalCandidate(
        symbol=symbol, candidate_id=_candidate_id(spec),
        method=str(spec.get("method", spec.get("family_combination", "UNKNOWN"))),
        direction=actual, signal_timestamp=str(result.get("signal_timestamp", "")),
        historical=HistoricalMetrics(**dict(historical)),
        current_setup_quality=float(result.get("current_setup_quality", 0.0)),
        has_current_signal=bool(result.get("has_current_signal", False)),
        candle_closed=bool(result.get("candle_closed", False)),
        rejection_reason=result.get("rejection_reason"),
    )

def build_symbol_candidate_pool(symbol: str, genomes: Iterable[StrategyGenome]) -> list[dict[str, Any]]:
    """Build family singles, pairwise hybrids and evolved genomes for one symbol.

    The pool deliberately does not call the legacy parity guard: these are research
    candidates and must be evaluated by a family-aware evaluator, never silently
    routed through the legacy SwingEngine adapter.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    baseline = StrategyGenome.baseline()
    seeds = list(genomes)
    # Ensure every signal family is represented, plus all pairwise family hybrids.
    for family in config.STRATEGY_FAMILIES:
        seeds.append(replace(baseline, signal_families=(family,), family_combination="SINGLE"))
    for left, right in combinations(config.STRATEGY_FAMILIES, 2):
        seeds.append(replace(
            baseline, signal_families=(left, right), family_combination="VOTE",
            min_family_agreement=2,
        ))
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for genome in seeds:
        genome.validate()
        methods = genome.entry_methods if "SWING_OB" in genome.signal_families else ("FAMILY_NATIVE",)
        for method in methods:
            one = replace(genome, entry_methods=(method,), symbols=(symbol,))
            payload = one.to_dict()
            for direction in DIRECTIONS:
                candidate_payload = {"genome": payload, "direction": direction}
                candidate_id = _canonical_hash(candidate_payload)[:20]
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                pool.append({
                    "candidate_id": candidate_id, "method": method,
                    "signal_families": list(one.signal_families),
                    "family_combination": one.family_combination,
                    "direction": direction, "genome": payload,
                })
    return sorted(pool, key=lambda item: (
        tuple(item["signal_families"]), item["family_combination"],
        item["method"], item["direction"], item["candidate_id"],
    ))

def run_symbol_cycle(
    *, symbol: str, cycle_id: str, candidates: Iterable[Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    checkpoint_path: str | Path, checkpoint_metadata: Mapping[str, Any],
    policy: ScoringPolicy = ScoringPolicy(),
) -> SelectionDecision:
    """Evaluate candidates with per-candidate atomic checkpoints and deterministic resume."""
    symbol = symbol.strip().upper()
    if not symbol or not cycle_id.strip():
        raise ValueError("symbol and cycle_id are required")
    specs = [dict(item) for item in candidates]
    ids = [_candidate_id(item) for item in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique within a symbol cycle")
    for spec in specs:
        direction = spec.get("direction")
        if direction is not None and str(direction).upper() not in DIRECTIONS:
            raise ValueError(f"unsupported candidate direction: {direction}")
        if "genome" in spec:
            StrategyGenome.from_dict(spec["genome"])
    pool_hash = _canonical_hash(specs)
    missing = {"code_hash", "config_hash", "dataset_hash"} - set(checkpoint_metadata)
    if missing:
        raise ValueError(f"checkpoint_metadata missing required keys: {sorted(missing)}")
    metadata = {**dict(checkpoint_metadata), "symbol": symbol, "cycle_id": cycle_id, "candidate_pool_hash": pool_hash}
    path = Path(checkpoint_path)
    completed: dict[str, dict[str, Any]] = {}
    if path.exists():
        restored = load_checkpoint(path, expected_metadata=metadata)
        completed = dict(restored["payload"].get("completed", {}))
    for index, spec in enumerate(specs):
        cid = _candidate_id(spec)
        if cid in completed:
            continue
        result = evaluator(spec)
        if not isinstance(result, Mapping):
            raise TypeError("candidate evaluator must return a mapping")
        _to_signal_candidate(symbol, spec, result)
        completed[cid] = dict(result)
        save_checkpoint(path, metadata=metadata, payload={
            "completed": completed, "status": "RUNNING",
            "next_candidate_id": _candidate_id(specs[index + 1]) if index + 1 < len(specs) else None,
        })
    signals = [_to_signal_candidate(symbol, spec, completed[_candidate_id(spec)]) for spec in specs]
    decision = select_cycle_signals(signals, policy=policy)[symbol]
    save_checkpoint(path, metadata=metadata, payload={
        "completed": completed, "status": "COMPLETED",
        "winner_candidate_id": decision.winner.candidate_id if decision.winner else None,
        "winner_method": decision.winner.method if decision.winner else None,
        "winner_direction": decision.winner.direction if decision.winner else None,
        "winner_score": decision.winner.final_score if decision.winner else None,
    })
    return decision
