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
