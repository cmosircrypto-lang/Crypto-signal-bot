# Dynamic Signal Selection Engine — Architecture Specification

## Status and scope

This is a research/design specification on an isolated branch. It does not alter the production signal engine, live scanner, order execution, or any existing trading logic. All experiments remain simulation-only; initial equity is $100 with compounding, and fees, slippage, and funding are set to zero until explicitly approved.

The objective is **cycle-by-cycle method selection**, not a permanent strategy assignment per symbol.

## 1. Core behavior

For every live scan cycle and every symbol:

1. Load the symbol's cached market data and incrementally update indicators/features.
2. Evaluate every enabled signal-generation method and permitted method/filter combination against the current closed-candle state.
3. For each method/combination, determine whether it has a valid current signal. A method with no current signal remains in the evaluation record but cannot win the cycle.
4. Score eligible candidates using historical validation evidence plus current-signal quality.
5. Select the highest-scoring eligible candidate for that symbol in this cycle only.
6. Emit at most one signal per symbol per cycle. If no candidate passes minimum quality, emit no signal.
7. Persist the cycle decision, all candidate scores, and progress/checkpoint state.

A symbol is not bound to a fixed strategy. For example, BNB may select a structure-break method in one cycle and an engulfing setup in the next. The previous winner does not receive permanent preference.

## 2. Candidate space

The candidate generator may combine, subject to bounded and testable search limits:

- Entry methods: DIRECT, ENGULFING, PIN_BAR, M15_STRUCTURE_BREAK, and future approved methods.
- Zone construction: body or wick.
- Zone interaction: first touch, maximum two touches, progressive touch rules.
- Trend/context filters: 1h, 4h, 1d, and approved combinations.
- BTC market-regime filters where enabled.
- Structure, confirmation, volatility, and momentum filters.
- Swing detection, order-block thresholds/lookbacks, ATR periods, and stop models once the underlying engine exposes those parameters safely.
- Risk and reward settings within approved bounds.

The genetic algorithm evolves candidate combinations and parameters. It is a discovery/optimization layer; the per-cycle selector is responsible for choosing among candidates that produce a valid current setup.

## 3. Two-stage scoring and selection

Do not rank a method solely by its in-sample profit or win rate.

### A. Historical method-quality score

Estimate each candidate's quality using chronological, leakage-safe historical evaluation. The evaluation should use rolling or walk-forward windows and retain a genuinely untouched final OOS segment.

Suggested normalized score components (weights are initial design defaults and must be calibrated using tests, not treated as proven):

- 25% risk-adjusted return / average R.
- 20% drawdown control.
- 15% profit factor, with safe handling for zero-loss samples.
- 15% consistency across rolling windows.
- 10% adequate trade count and statistical confidence.
- 10% out-of-sample validation evidence (only after candidate generation/selection is frozen for that validation stage).
- 5% simplicity/stability penalty for overly complex or parameter-fragile candidates.

Hard eligibility gates take precedence over the weighted score: minimum trade count, maximum drawdown, minimum validation quality, and no leakage/parity failures. Thresholds should be configured and versioned.

### B. Current-cycle signal score

For candidates that produce a valid current signal, combine:
- the candidate's historical method-quality score;
- current setup quality (entry/zone validity, confirmation, timeframe alignment, and distance/volatility sanity);
- freshness and closed-candle confirmation;
- risk/reward feasibility under the configured simulation constraints.

The selector chooses the highest-scoring eligible current signal for that symbol. Scores must be comparable across methods and symbols through normalization. Store component scores as well as the final score for auditability.

A high historical score alone must never create a signal when the method has no valid current setup. If all candidates fail eligibility or minimum-score gates, return NO_SIGNAL.

## 4. Anti-lookahead and validation rules

- Use closed candles only.
- Align 1h/4h/1d features causally to the 15m decision timestamp; never expose a higher-timeframe candle before it has closed.
- Split data chronologically into train, validation, and final OOS. Use walk-forward/rolling validation where practical.
- Never use final OOS outcomes to mutate genes, tune weights, choose thresholds, or select the winning method.
- Fit/normalize/derive any statistical feature using data available at the decision timestamp only.
- Preserve the existing same-candle SL/TP rule (SL_FIRST) unless a separately approved experiment changes it.
- Keep the baseline simulation constraints: $100 starting equity, compounding, 6 total positions, 3 per direction, at most 1 open position per symbol, and zero costs for now.

## 5. Checkpointing and resumability

Checkpointing is required at multiple granularities, not only after an entire symbol finishes.

### Checkpoint levels

1. **Dataset/feature checkpoint:** cached source-data identity, last processed candle timestamp, and indicator/feature state.
2. **Symbol checkpoint:** symbol status, completed stages, and the next candidate/method index.
3. **Candidate checkpoint:** candidate/genome ID, parameter fingerprint, evaluation window, completed folds, metrics, and score components.
4. **Cycle checkpoint:** cycle ID/time, per-symbol candidate evaluations, winner/no-winner decision, and emitted-signal record.

### Required checkpoint metadata

- Schema version and checkpoint format version.
- Git commit SHA and code/configuration hash.
- Dataset identity/hash, symbol, timeframe set, candle range, and last closed-candle timestamp.
- Candidate/genome ID and canonical parameter hash.
- Random seed and evolution settings (where GA is involved).
- Completed stage/fold/candidate identifiers and pending work.
- Creation timestamp and integrity checksum.

### Resume behavior

- Save checkpoints atomically (write temporary file, flush/close, then replace) and retain a last-known-good checkpoint.
- Persist completed candidate results immediately or in small batches; do not wait for all 76 symbols.
- On restart, validate schema, code/config hash, dataset hash, and candidate fingerprint before reuse.
- Resume only unfinished candidates/folds. Never repeat completed valid work.
- If data, code, or relevant configuration changes, invalidate only affected checkpoints and explain why; do not silently reuse stale results.
- Make candidate evaluation deterministic from its seed and inputs.
- Record explicit states such as PENDING, RUNNING, COMPLETED, FAILED_RETRYABLE, FAILED_FINAL, and INVALIDATED.
- Upload checkpoint artifacts frequently enough to survive a GitHub Actions job timeout. Prefer symbol/candidate shards and a resumable matrix or staged workflow over one monolithic job.

## 6. Runtime performance

Running a full genetic evolution from scratch on every live scan would be unnecessarily expensive. Separate the work into:

- Offline/periodic discovery: evolve and validate a candidate pool, with resumable checkpoints.
- Live cycle evaluation: use cached data/features and evaluate the current signal from the approved candidate/method pool; calculate cycle-specific scores and select the winner.
- Scheduled refresh: re-run discovery when new data accumulates or a configured refresh condition is met, without blocking ordinary live cycles.

This preserves the user's requirement that methods are re-tested and re-scored on every live cycle while avoiding repeated full-history downloads and full genetic evolution on every cycle. The exact refresh cadence is a configurable decision to validate experimentally.

## 7. Parallelism and fault isolation

- Symbols and independent candidate evaluations should be shardable and parallelizable.
- A failure on one symbol must not discard completed work for other symbols.
- Each worker writes isolated checkpoint/result files; a coordinator merges them deterministically.
- Merge must be idempotent and reject conflicting duplicate results.
- Resource limits and concurrency must be configurable to avoid API throttling or runner memory exhaustion.

## 8. Audit output

For each symbol and cycle, persist:
- all methods/combinations considered;
- whether each had a current signal;
- historical quality and current setup score components;
- eligibility/rejection reasons;
- selected method and final score, or NO_SIGNAL;
- candidate/version/data fingerprints;
- timestamps and checkpoint references.

Produce aggregate reports by cycle, symbol, method, and validation window. This allows us to detect method instability, regime dependence, and selection bias.

## 9. Implementation gates

Implement in isolated research steps:

1. Add schema/config and unit tests for cycle records, candidate scores, and checkpoint compatibility.
2. Add deterministic checkpoint/resume tests, including interruption mid-symbol and mid-candidate.
3. Add causal higher-timeframe alignment and explicit leakage tests.
4. Add candidate scoring/eligibility and deterministic per-symbol cycle selection tests.
5. Add a small cached-data smoke run, then resumable full 76-symbol validation.
6. Compare against the existing SwingEngine baseline for parity wherever the candidate uses legacy behavior.
7. Only after research results and explicit user review should any production/live integration be proposed.

No production integration, live order placement, or changes to the existing signal logic are authorized by this specification.


## 2.1 Direction is an independent evolutionary dimension

LONG and SHORT must not be generated as a single undirected strategy and assigned a direction only after optimization. For every symbol and every eligible method/genome, create separate LONG and SHORT candidate evaluations and maintain direction-specific historical metrics, rolling-validation folds, mutation/selection outcomes, and current-setup quality. A LONG candidate's trades and metrics must never be pooled into its SHORT counterpart, or vice versa. Each candidate's evaluator must confirm that its result direction matches its declared direction; mismatches are invalid, not silently relabeled.

The final per-symbol cycle selector may compare eligible LONG and SHORT candidates on a common normalized score scale and choose the best single signal for that symbol. This final comparison does not replace the independent direction-specific evolution and validation. If neither side passes the gates, return NO_SIGNAL. Portfolio-level limits (including the existing three-open-per-direction cap) are applied after per-symbol selection and must not contaminate candidate fitness with outcomes from another direction.

## 2.2 Additional strategy families to research from main.py

The current `main.py` on the repository's `main` branch was reviewed as a source of candidate strategy families. It contains three distinct signal logics (the three managed variants reuse those same logics; they are not three additional strategies):

1. **EMA Cross PRO (`analyze_cross`)** — recent 4h EMA9/EMA21 cross; 4h ADX, EMA gap, EMA slope and EMA21/EMA50 alignment; 1h trend/ADX/gap and volume confirmation; 15m RSI band; ATR-based stop/target multipliers and a weighted setup score. LONG and SHORT have separate RSI bands and directional checks.
2. **BTC-aligned 1h pullback (`analyze_btc_aligned`)** — BTC 1d/4h directional agreement and strength/extension checks; symbol 4h bias agreement; 1h EMA21 pullback/touch and reclaim/rejection candle; direction-specific RSI and volume confirmation; ATR-based SL/TP. The code also has a fallback to symbol 4h bias when BTC context is unavailable, which must be represented as a separate explicit mode if tested, never silently conflated with BTC-filtered behavior.
3. **SMC Winner (`analyze_smc_winner`)** — 4h liquidity sweep of prior lookback high/low followed by close back inside; EMA21/EMA50 trend confirmation; wick/ATR stop distance bounds; ATR target and minimum RR; volume and direction-specific RSI extreme checks; score based on trend, volume, wick and RR.

The managed bot variants (`v0a`, `v60a`, `v70a`) reuse these three signal functions and add sizing/recovery/account-state behavior. Only their signal-generation logic is a candidate source here. Their 3% risk, 2% SL sizing, recovery-loss accounting, Telegram workflow, signal persistence, and bot-specific cooldown/dispatch rules are not strategy genes and must not be copied into this research engine.

These three strategy families should be ported as isolated, deterministic research evaluators against the cached OKX dataset, not imported from `main.py` at runtime. Their current implementation uses Toobit/CCXT data access, different symbol coverage and timeframes, live-time candle trimming, external messaging/state, and some execution-time ticker checks. Reusing the file directly would couple the research engine to production side effects and create exchange/timeframe/causality mismatches.

Before admitting them to genetic evolution:
- Port the signal rules only, preserving a documented baseline parameter set.
- Map every rule to causal 15m/1h/4h/1d bars from the cached OKX dataset; do not assume a current live candle or future higher-timeframe close is available.
- Recompute historical signals using only information available at each decision timestamp.
- Build direction-specific LONG and SHORT candidate populations and metrics.
- Add parity/fixture tests for known candles and expected signal/no-signal decisions, plus SL/TP level calculations.
- Test each family standalone against the current SwingEngine and against each other before combining their genes with existing DIRECT/ENGULFING/PIN_BAR/M15_STRUCTURE_BREAK candidates.
- Keep all bot-specific capital management, recovery, notification, file I/O, and order/state handling outside the strategy evaluator.

The three main.py strategies are promising additional candidate families, but they are not yet wired into the genetic evaluator and no performance claim is made about them.


## 2.3 Composable genetic strategy families (expanded gene space)

The genome is a composition of independently evolvable signal families and their parameter genes, not a hard-coded mapping from a coin to one strategy. Every symbol receives the same research family catalogue; each symbol/cycle can rank different evolved combinations. The initial catalogue is:

- `SWING_OB`: existing DIRECT, ENGULFING, PIN_BAR and M15_STRUCTURE_BREAK methods, with zone/touch/structure genes.
- `EMA_CROSS_PRO`: recent-cross window, fast/slow EMA periods, 4h/1h ADX thresholds, volume multiplier, EMA slope threshold, ATR stop and target multipliers.
- `BTC_ALIGNED_PULLBACK`: BTC regime/timeframe mode, minimum EMA gap in ATR, EMA touch tolerance, maximum extension, volume threshold, ATR stop and target.
- `SMC_LIQUIDITY_SWEEP`: sweep lookback, stop buffer, min/max stop distance, ATR target, minimum RR and minimum volume ratio.
- New research families: `MOMENTUM_BREAKOUT` (breakout lookback/buffer, ADX and volume), `MEAN_REVERSION` (directional RSI bounds and z-score threshold), and `VOLATILITY_SQUEEZE` (Bollinger period/width percentile and expansion threshold).

The parameter domains are bounded in `config.py` and validated/serialized by `StrategyGenome`. The main.py values are included as baseline points in those domains, not treated as optimal values. Other genes may mutate away from those baselines.

Combination modes:
- `SINGLE`: exactly one family.
- `ANY`: any active family may independently produce the candidate signal.
- `ALL`: all active families must agree on direction and signal timestamp/window.
- `VOTE`: at least `min_family_agreement` families must agree.

The cycle-pool builder guarantees baseline coverage of every family, every legacy Swing entry method, and pairwise family hybrids, in both LONG and SHORT. Evolved genomes may add further combinations and parameter variants. The pairwise seed set is a coverage floor, not a restriction on genetic search; mutation/crossover may form larger combinations within configured complexity limits.

**Important implementation boundary:** schema and candidate-pool support do not themselves implement the family signal evaluators or the genetic mutation/crossover operators for every new gene. The existing SwingEngine adapter still cannot evaluate the new families or most newly exposed parameter genes. A family-aware causal OKX evaluator must be built and fixture-tested before any of these candidates are treated as measured or eligible. No candidate is currently proven profitable by merely appearing in the pool.
