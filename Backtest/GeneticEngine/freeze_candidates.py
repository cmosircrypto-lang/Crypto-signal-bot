"""Freeze the eligible champions from one completed genetic run.

This utility copies the exact serialized genomes; it never mutates or ranks them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .genome import StrategyGenome

EXPECTED_SYMBOLS = (
    "ADA-USDT-SWAP", "BNB-USDT-SWAP", "BTC-USDT-SWAP",
    "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP",
)
EXPECTED_DIRECTIONS = ("LONG", "SHORT")


def build_frozen_manifest(
    artifact_root: str | Path, *, source_run_id: int, source_commit_sha: str,
) -> dict[str, Any]:
    root = Path(artifact_root)
    summaries = sorted(root.rglob("genetic_family_summary_*.json"))
    if len(summaries) != 12:
        raise ValueError(f"Expected 12 source direction summaries, found {len(summaries)}")

    seen: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        symbol = str(payload["symbol"]).upper()
        direction = str(payload["direction"]).upper()
        key = (symbol, direction)
        if symbol not in EXPECTED_SYMBOLS or direction not in EXPECTED_DIRECTIONS:
            raise ValueError(f"Unexpected source direction: {key}")
        if key in seen:
            raise ValueError(f"Duplicate source direction: {key}")
        seen.add(key)

        result = payload["result"]
        champion = result.get("champion") or result.get("best_exploratory")
        if not isinstance(champion, dict):
            raise ValueError(f"Missing champion payload for {key}")
        metrics = champion.get("metrics") or {}
        genome_payload = champion.get("genome")
        if not isinstance(genome_payload, dict):
            raise ValueError(f"Missing serialized genome for {key}")

        if champion.get("eligible") is True:
            genome = StrategyGenome.from_dict(genome_payload)
            if genome.symbols != (symbol,):
                raise ValueError(f"Genome symbol scope mismatch for {key}: {genome.symbols}")
            candidates.append({
                "candidate_id": f"{symbol}__{direction}",
                "symbol": symbol,
                "direction": direction,
                "genome": genome.to_dict(),
                "historical_reference_metrics": metrics,
                "source_summary": path.name,
            })
        else:
            excluded.append({
                "candidate_id": f"{symbol}__{direction}",
                "symbol": symbol,
                "direction": direction,
                "eligible": False,
                "reason": "Champion was not eligible in the source genetic run.",
                "historical_reference_metrics": metrics,
                "source_summary": path.name,
            })

    expected = {(s, d) for s in EXPECTED_SYMBOLS for d in EXPECTED_DIRECTIONS}
    if seen != expected:
        raise ValueError(f"Source directions incomplete: missing={sorted(expected-seen)}")
    if len(candidates) != 11 or len(excluded) != 1:
        raise ValueError(
            f"Expected exactly 11 eligible and 1 excluded champions; "
            f"found {len(candidates)} eligible and {len(excluded)} excluded"
        )
    if [(x["symbol"], x["direction"]) for x in excluded] != [("BNB-USDT-SWAP", "LONG")]:
        raise ValueError("Unexpected excluded direction; source eligibility changed")

    candidates.sort(key=lambda x: (x["symbol"], x["direction"]))
    excluded.sort(key=lambda x: (x["symbol"], x["direction"]))
    return {
        "schema_version": 1,
        "frozen": True,
        "selection_rule": (
            "Copy only source champions whose eligible flag is true. "
            "No mutation, evolution, retraining, parameter selection, or reselection."
        ),
        "source": {
            "repository": "cmosircrypto-lang/Crypto-signal-bot",
            "branch": "experiment/dynamic-signal-selection-design",
            "run_id": int(source_run_id),
            "commit_sha": source_commit_sha,
        },
        "window": {
            "start_utc": "2026-11-17T00:00:00Z",
            "end_exclusive_utc": "2026-11-22T22:00:00Z",
            "timeframe": "15m",
        },
        "candidates": candidates,
        "excluded": excluded,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--source-run-id", required=True, type=int)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = build_frozen_manifest(
        args.artifact_root,
        source_run_id=args.source_run_id,
        source_commit_sha=args.source_commit_sha,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[FROZEN] eligible={len(manifest['candidates'])}; "
        f"excluded={len(manifest['excluded'])}; source_run={args.source_run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
