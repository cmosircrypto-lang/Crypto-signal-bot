# Genetic Strategy Engine

This package is an isolated research layer over the existing SwingEngine.
It is simulation-only and does not submit exchange orders.

## Fixed baseline

- OKX USDT linear SWAP historical data: 76 symbols
- 15m execution; 1h/4h and causally derived 1d analysis
- Initial equity: $100
- Compounding remains inside the existing execution engine
- Baseline fees and slippage: zero
- Portfolio limits: 6 total, 3 per direction, 1 per symbol
- Closed candles only; same-candle SL/TP collision resolves to SL first

## First implementation slice

- config.py: immutable research constraints and bounded search domains
- genome.py: versioned candidate representation with validation and serialization
- evaluator.py: compatibility adapter that calls the existing SwingEngine

The adapter intentionally refuses non-default swing/OB and ATR-stop parameters for now.
The current SwingEngine hardcodes swing 5/5 in signal generation and the
OB detector defaults to threshold 1.0, ATR 14 and lookback 10. Its stop is
currently derived from the OB wick; ATR-based stops are not yet implemented.
Those parameters and stop models must be wired through default-preserving,
parity-tested changes before the genetic search is allowed to mutate them.

## Tests

Run from the repository root:

    python -m unittest Backtest.GeneticEngine.tests.test_genome

This initial test module validates the genome contract and baseline defaults.
A full data-backed parity test (signals, trades, equity and trade ledger) is
a release gate before any genetic population is evaluated. It must compare
the adapter against direct calls to the existing runner on the same cached
dataset and contract specifications.


## Full cached-data parity gate

The opt-in GitHub Actions input `run_genetic_data_parity` downloads the same
cached OKX release used by the SwingEngine and runs:

    python Backtest/GeneticEngine/tests/test_cached_data_parity.py

This gate validates all 76 symbols across 15m/1h/4h, loads current public OKX
USDT-linear-SWAP contract specifications, and compares the direct SwingEngine
pipeline with the Genetic Engine adapter using exact frame/ledger equality.
It checks prepared order blocks, entry signals, the complete trade ledger, and
the backtest summary (including equity and drawdown metrics). It does not run
any genetic population or submit orders. It requires network access to the
public OKX instruments endpoint and may take substantial time because both
pipelines are evaluated across the full cached dataset.


## Dynamic cycle-by-cycle signal selection (research implementation)

The isolated research branch also contains:

- `dynamic_selector.py`: scores each candidate using historical training/rolling-validation metrics plus current setup quality, then selects at most one valid current signal per symbol for that cycle. It is stateless across cycles, so the winning method may change on the next scan. No eligible setup means `NO_SIGNAL`.
- `checkpoint.py`: atomically saves JSON checkpoints with SHA-256 integrity verification and strict metadata compatibility checks (for example code/config and dataset hashes).
- `tests/test_dynamic_selection.py`: unit tests for ranking, no-signal gates, deterministic ties, checkpoint round-trip, stale metadata and tamper detection.

The initial score weights and eligibility thresholds are research defaults, not proven trading parameters. Historical scoring consumes training/rolling-validation metrics only; the final untouched OOS set must not be used to evolve candidates, tune weights, or select a method. Current-candle eligibility requires a closed candle. The modules are not wired into the production or live signal pipeline and do not place orders.

Run the focused tests with:

    python -m unittest Backtest.GeneticEngine.tests.test_dynamic_selection


### Resumable per-symbol cycle runner

`dynamic_cycle_runner.py` evaluates a supplied candidate pool for one symbol and one scan-cycle ID. It checkpoints after every completed candidate, validates the candidate-pool fingerprint plus caller-supplied code/config and dataset hashes, resumes without re-evaluating completed candidates, and stores the cycle winner (or no-signal result). Each symbol/cycle should use its own checkpoint path. The candidate evaluator is injected by the research adapter; this keeps the runner decoupled from the legacy SwingEngine and allows a future genetic candidate generator to supply its approved candidate pool.

The current runner is an orchestration primitive, not yet a full 76-symbol live integration. A caller must provide candidates and an evaluator that returns historical training/rolling-validation metrics and the current closed-candle setup. Do not feed final OOS outcomes into those metrics. The CI tests include a simulated interruption and verify that completed candidates are not repeated.


The runner's `build_symbol_candidate_pool` helper expands each genome into one-method candidates scoped to the requested symbol. It uses the existing adapter's compatibility guard, so unsupported legacy parameters are rejected rather than silently treated as active genes. Checkpoint metadata must include `code_hash`, `config_hash`, and `dataset_hash`; the runner also fingerprints the symbol, cycle ID, and complete candidate pool. The checkpoint records the next candidate cursor after each completed evaluation.


The per-symbol pool builder also injects a baseline candidate for every configured entry method, so a small or skewed genetic population cannot accidentally omit an enabled method from a scan-cycle comparison. Evolved genomes add their distinct filter/zone/touch/risk/RR combinations alongside this guaranteed method coverage.


## Direction-specific evolution and main.py strategy sources

The dynamic candidate pool now expands each method/genome into separate LONG and SHORT candidate specifications. Evaluator results must match the declared direction. Historical metrics and evolution must remain direction-specific; only after independent evaluation may the cycle selector compare both sides and choose at most one signal for a symbol.

The three distinct signal families in the repository's `main.py` are documented as future research candidates: EMA Cross PRO, BTC-aligned 1h pullback, and SMC Winner. The three managed bot variants reuse those same signal functions and are not additional strategies. Only signal-generation rules may be ported. Position sizing, recovery, Telegram, persistence, cooldown/dispatch and live exchange access are excluded. The strategies are not yet wired into the genetic evaluator; they require causal OKX data adapters, fixtures and parity tests first.


## Composable family genome (research schema)

The genome now carries parameter genes for seven signal families: existing Swing/OB methods; the three signal families inspected in main.py (EMA Cross PRO, BTC-aligned pullback, SMC liquidity sweep); and three additional research families (momentum breakout, mean reversion, volatility squeeze). Main.py's constants are baseline points inside bounded search domains; they are not frozen settings or assumed winners.

Families can be combined using SINGLE, ANY, ALL, or VOTE with a configurable agreement threshold. The candidate-pool builder provides baseline coverage of all families, pairwise hybrids, the four Swing entry methods, and independent LONG/SHORT candidates. The runner validates serialized genomes and checkpoints each candidate.

**This is a genome/schema and orchestration expansion, not yet a completed family-aware signal engine.** The existing legacy SwingEngine adapter still does not consume the new family genes. A deterministic, causal OKX implementation for each family, parameter injection, mutation/crossover coverage, and fixture/parity tests remain required before performance runs or live-cycle use. All work remains isolated to this research branch.


CI now compiles the expanded config/genome modules and runs the strategy-family genome tests alongside selector, checkpoint and legacy contract tests.


## Genetic mutation and crossover

`evolution.py` adds deterministic seeded `random_genome`, `mutate_genome`, and `crossover_genomes` operators. They can vary the existing Swing genes, all three main.py-derived parameter groups, and the additional momentum/mean-reversion/squeeze families; family lists and combination modes are genes too. Dependent constraints (fast EMA below slow EMA, minimum stop below maximum stop, RSI low below RSI high, and valid family agreement) are repaired before a child is accepted. These are tested operators and genome sampling, not yet a complete population evolution/evaluation loop; family signal evaluators remain a separate gate.


## Causal family signal evaluators

`family_signals.py` now provides closed-candle evaluators for EMA Cross PRO,
BTC-aligned pullback, SMC liquidity sweep, momentum breakout, mean reversion,
and volatility squeeze. The three families derived from the active strategies
in the repository's `main.py` use their corresponding baseline concepts and
genome parameters. Swing/Order Block remains an explicit adapter to a validated
closed-candle SwingEngine signal; this module does not pretend a simplified
proxy is parity-equivalent to the existing SwingEngine.

The evaluator supports SINGLE, ANY, ALL, and VOTE. Hybrid signals require
direction agreement; ALL requires every selected family, while VOTE requires
`min_family_agreement`. Hybrid stop and target distances use medians of the
agreeing families. LONG and SHORT are evaluated in separate calls.

`make_cycle_evaluator` adapts current family signals to the resumable cycle
runner callback contract. Historical metrics must still be supplied by an
explicit historical backtest callback using training/validation folds only.
No synthetic or fabricated historical performance is accepted. The current
signal evaluator itself is not a substitute for the historical trade simulator.

All input frames must be pre-truncated to fully closed candles at the cycle
cutoff. The module does not fetch data, and the untouched final OOS segment
must remain outside the historical evaluator and all candidate selection.


## Walk-forward historical evaluation and genetic population loop

The current research branch now includes:

- `historical_evaluator.py`: constructs complete UTC BTC daily candles from 24 contiguous 1h bars; makes decisions only from 15m/1h/4h candles whose close time is at or before the decision time; simulates entries at the next 15m open; and applies SL-first resolution when both SL and TP are touched in one candle.
- Walk-forward validation windows occupy the development portion of the series. The final 15% is excluded before any candidate metrics are produced. Fold metrics are aggregated from closed trades; no final OOS performance is used for fitness.
- `genetic_runner.py`: executes independent populations for each symbol and each direction. It uses seeded initialization, family-single and pairwise-hybrid seeds, tournament selection, elitism, crossover, mutation, phenotype deduplication and per-candidate atomic checkpoints.
- Checkpoint compatibility fingerprints include code/config/data identity, direction, random seed, population/generation settings, seed-genome fingerprint and fitness gates. A changed dataset or research configuration is rejected rather than silently resumed.
- `run_genetic_research.py`: command-line entry point that loads and validates the cached 76-symbol dataset, fingerprints it, runs independent LONG/SHORT populations, and writes a JSON research summary. It is simulation-only.

Example smoke run from the repository root:

    python -m Backtest.GeneticEngine.run_genetic_research --profile smoke --symbols BNB-USDT-SWAP --population 8 --generations 2

Run the full configured evolution only after a smoke run and runtime estimate:

    python -m Backtest.GeneticEngine.run_genetic_research --profile main

The smoke defaults are intentionally small. The main profile is 100 individuals × 30 generations per symbol and direction, and a 76-symbol run is computationally substantial. Run by symbol batches and preserve the generated checkpoints; do not start a full 76-symbol main profile without estimating runner time and storage first.

The walk-forward simulator currently uses theoretical base-quantity sizing, compounding equity, one open trade per symbol within each independent symbol/direction evaluation, zero fees and zero slippage, and a forced close at each validation-fold boundary for any still-open position. This per-symbol genetic fitness evaluation is not the final multi-symbol portfolio backtest. The selected champions must subsequently be tested together under the existing portfolio limits (6 total, 3 per direction, 1 per symbol) and on a final untouched OOS segment exactly once, after all research choices are frozen.

Swing/Order Block signals require an explicit closed-candle `swing_signal_provider`; without one, the family correctly emits no historical trades rather than substituting a simplified proxy. The three main.py-derived families and the additional families use the new causal evaluators. No live signal pipeline or order execution is connected.
