# Swing Engine — isolated OKX perpetual backtest with BTC trend filters

This engine is simulation-only. It does not submit live orders and does not
modify the legacy scanner or live trading code.

## Dataset layout

The loader expects validated OKX archives at:

- `Backtest/data/okx/15m/*.csv.gz`
- `Backtest/data/okx/1h/*.csv.gz`
- `Backtest/data/okx/4h/*.csv.gz`

Each file must contain `ts,open,high,low,close,volume`. The `ts` value is
the candle OPEN time in Unix milliseconds. The default run validates all 76
symbols across all three timeframes.

## Run the 36-scenario BTC-filtered backtest

From the repository root:

```bash
pip install pandas numpy
python Backtest/SwingEngine/backtest_runner.py
```

By default, the runner downloads public OKX instrument specifications and
applies linear USDT SWAP contract lot-size, minimum-size, and tick-size rules.
No API key is required. If a calculated order is below the exchange minimum,
the trade is skipped and counted; the runner never increases risk to force
an order.

For theoretical base-quantity sizing without exchange lot/minimum constraints:

```bash
python Backtest/SwingEngine/backtest_runner.py --legacy-sizing
```

The full runner now tests three BTC trend filters at entry time: 1d+4h,
1d+1h, and 1d+4h+1h. A signal is accepted only when BTC is aligned with its
direction on every timeframe required by that filter. Neutral and unavailable
states are rejected. Daily BTC candles are aggregated from exactly 24
contiguous UTC 1h candles. Only candles fully closed by the signal timestamp
are used; no future candle can influence the entry. The optional
`--require-4h-alignment` remains off by default and is separate from these BTC
filters.

## Test matrix

Initial capital is $100. Every run uses the same signals, entry, and stop
placement, with zero fees, zero slippage, and zero funding:

| Risk per trade | RR values |
|---|---|
| 0.05% | 1.5, 2.0, 2.5 |
| 0.5% | 1.5, 2.0, 2.5 |
| 1.0% | 1.5, 2.0, 2.5 |
| 1.5% | 1.5, 2.0, 2.5 |

Each risk/RR pair is replayed independently under all three BTC filter modes,
for 3 × 4 × 3 = 36 chronological portfolio simulations.

The stop is placed at the far wick edge of the source 1h Order Block:
below the lower wick for BUY and above the upper wick for SELL. The target is
calculated from the entry-to-stop distance multiplied by the selected RR.

## Execution and portfolio rules

- Maximum 6 simultaneous positions.
- Maximum 3 BUY and maximum 3 SELL.
- One open position per symbol; symbols with an open position are not rescanned.
- Entries occur at the close of the signal 15m candle.
- Exit checks begin on the next 15m candle, never on the entry candle.
- If SL and TP are both touched in one 15m candle, SL is assumed first.
- Risk is calculated from current equity and is not multiplied by leverage.
- No fees, slippage, or funding are applied at this stage.
- Positions still open at the dataset end are reported separately and are not
  force-closed at an invented price.

## Outputs

Files are written to `Backtest/results/swing_engine/`:

- `summary_btc_filters_36_scenarios.csv` and `summary_btc_filters_36_scenarios.json`
- One `trades_*.csv` ledger for each of the 36 scenarios

The summary includes trade count, win rate, profit factor, average R,
maximum realized-equity drawdown, ending equity, open positions, and skipped
trade counts.

## Post-backtest diagnostics

After all 36 scenarios finish, the runner also preserves signal attribution
in each closed-trade ledger: entry method, source OB ID/timestamps, BOS time,
touch number/score, 4h trend label, and the actual stop/target used. Separate
CSV ledgers record skipped signal candidates (including the reason, method,
symbol, and scenario) and positions still open at the end of the dataset.

The analysis-only `analyze_results.py` module then creates:
- Performance by entry method (DIRECT, ENGULFING, PIN_BAR,
  M15_STRUCTURE_BREAK).
- Performance by symbol, symbol + method, entry month, and exit month.
- Performance by BUY/SELL direction, touch number, and exit reason.
- Skipped-signal counts by reason, method, and symbol.
- A cross-scenario ranking by Profit Factor and net PnL.

These reports are derived from the existing simulated trades. They do not
change signal generation, execution order, sizing, stop/target logic, or
backtest results. The outputs are saved beside the 36 trade ledgers under
`Backtest/results/swing_engine/`.

## No-lookahead rule

A candle is available to a decision only when its close time has passed:
`candle_open_timestamp + timeframe_duration <= decision_timestamp`.


## Live progress and stall diagnosis

The runner reports progress from the beginning of the job, not only during
the nine portfolio simulations:

- Dataset validation/loading: completed symbols out of 76, with a 30-second
  heartbeat while a compressed file is being read.
- Order Block detection: current symbol and scanned 1h candle count.
- Signal preparation: current symbol, Order Block scan count, and 15m candle
  scan count. Long inner scans emit progress callbacks every 50,000 candles.
- Timeline construction: symbols indexed for the global 15m close timeline.
- Each of the 36 scenarios: progress approximately every 5%, active UTC
  month, elapsed time, and estimated remaining time.
- Saving: completed trade ledgers out of 36.

A `[HEARTBEAT]` line is emitted every 30 seconds and identifies the exact
stage and current symbol/operation. If the same operation remains active for
a long time, the heartbeat continues to show that it is alive but not
advancing its sub-step counter; this helps distinguish a slow nested scan
from a silent or stalled process. If no sub-step callback arrives for two
minutes, the runner also prints `[STALL WARNING]` with the current stage and
operation. This warning does not kill or alter the run. The heartbeat is
diagnostic and does not change signal rules, event ordering, position sizing,
or calculated results.
