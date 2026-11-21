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
