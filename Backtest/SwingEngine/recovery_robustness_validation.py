"""Independent chronological OOS robustness tests for serialized recovery."""
from __future__ import annotations
import json, os, pickle, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from recovery_cycle_resumable import CUTOFF_UTC, INITIAL_CAPITAL, _simulate
from data_loader import TIMEFRAME_MS
from okx_contract_specs import OKXContractSpecs

RISK_CAPS=(3.0,5.0)
PERIODS=3
ATR_MULTIPLE=2.0
REWARD_RATIO=2.0

def _segment_ranges(start_ms, end_exclusive_ms, count=PERIODS):
    if count<1 or end_exclusive_ms<=start_ms:
        raise ValueError("A positive interval and segment count are required")
    width=end_exclusive_ms-start_ms
    b=[start_ms+width*i//count for i in range(count+1)]
    return [(b[i],b[i+1]) for i in range(count)]

def _dataset_time_bounds(data):
    bounds=[]
    for item in data.values():
        f=item.frames["15m"]
        if not f.empty:
            c=f["ts"].astype("int64")+TIMEFRAME_MS["15m"]
            bounds.extend((int(c.min()),int(c.max())))
    if not bounds: raise ValueError("Prepared OOS cache contains no 15m candles")
    return min(bounds),max(bounds)+1

def _slice_period(data,signals,start_ms,end_ms):
    pd_data,pd_signals={},{}
    candles=signal_count=symbol_candles=symbol_signals=0
    for symbol,item in data.items():
        f=item.frames["15m"]
        close=f["ts"].astype("int64")+TIMEFRAME_MS["15m"]
        chosen=f.loc[(close>=start_ms)&(close<end_ms)].reset_index(drop=True)
        pd_data[symbol]=SimpleNamespace(symbol=symbol,frames={"15m":chosen})
        candles+=len(chosen); symbol_candles+=int(not chosen.empty)
        sf=signals[symbol]
        if sf.empty: ss=sf.copy()
        else:
            ts=sf["signal_ts"].astype("int64")
            ss=sf.loc[(ts>=start_ms)&(ts<end_ms)].reset_index(drop=True)
        pd_signals[symbol]=ss
        signal_count+=len(ss); symbol_signals+=int(not ss.empty)
    return pd_data,pd_signals,{"segment_candle_count":candles,
        "segment_signal_count":signal_count,
        "segment_symbols_with_candles":symbol_candles,
        "segment_symbols_with_signals":symbol_signals}

def _utc(ms):
    return datetime.fromtimestamp(ms/1000,tz=timezone.utc).isoformat()

def run():
    cache=Path(os.environ.get("RECOVERY_PREPARED_CACHE",
        "Backtest/results/recovery_adaptive_matrix/prepared_oos_cache.pkl"))
    if not cache.exists(): raise FileNotFoundError(f"Prepared OOS cache not found: {cache}")
    out=Path(os.environ.get("RECOVERY_ROBUSTNESS_OUTPUT_DIR",
        "Backtest/results/recovery_robustness"))
    out.mkdir(parents=True,exist_ok=True)
    with cache.open("rb") as h: p=pickle.load(h)
    data,signals=p["test_data"],p["test_signals"]
    obs,btc_frames=p["obs"],p["btc_frames"]
    selected=p["sets"].get("all_76",[])
    if len(selected)!=76: raise ValueError(f"Expected 76 symbols, found {len(selected)}")
    data={s:data[s] for s in selected}; signals={s:signals[s] for s in selected}
    start,end=_dataset_time_bounds(data)
    ranges=_segment_ranges(start,end,PERIODS)
    specs=OKXContractSpecs(); specs.load()
    rows=[]; started=time.monotonic(); total=PERIODS*len(RISK_CAPS); run_no=0
    for period,(a,b) in enumerate(ranges,1):
        d,s,counts=_slice_period(data,signals,a,b)
        if counts["segment_candle_count"]==0: raise ValueError(f"Period {period} has no candles")
        for cap in RISK_CAPS:
            run_no+=1
            label=f"period_{period}_{_utc(a)[:10]}_to_{_utc(b-1)[:10]}_portfolio_{int(cap)}pct"
            cfg={"objective":"initial_target_profit","sizing_method":"exact_recovery",
                "max_recovery_trades":None,"portfolio_open_risk_cap_percent":cap,
                "max_total_positions":1,"max_buy_positions":1,"max_sell_positions":1,
                "single_position_global":True,"keep_cycle_when_risk_blocked":True,
                "smart_balance_cap":False}
            def report(done,count,ts):
                if done==count or done%max(1,count//10)==0:
                    print(f"[PROGRESS] {run_no}/{total} {label}: {done}/{count}",flush=True)
            summary,trades,cycles,skips=_simulate(
                d,s,obs,specs,btc_frames,ATR_MULTIPLE,REWARD_RATIO,"all_76",cfg,report)
            summary.update({"scenario":label,"period_index":period,
                "segment_start_utc":_utc(a),"segment_end_utc_exclusive":_utc(b),
                "segment_duration_days":(b-a)/86400000,**counts,
                "segment_symbols":len(selected),"segment_initial_equity":INITIAL_CAPITAL,
                "periods_are_independent":True,
                "recovery_method":"unlimited_until_tp_or_equity_wipeout",
                "recovery_objective":"initial_target_profit","recovery_sizing":"exact_recovery",
                "portfolio_open_risk_cap_percent":cap,
                "max_simultaneous_positions_limit":1,"single_position_global":True,
                "atr_multiple":ATR_MULTIPLE,"reward_ratio":REWARD_RATIO,
                "fee_rate":0.0,"slippage_rate":0.0,"funding_cost":0.0,
                "liquidation_definition":"conservative equity-wipeout proxy; not exchange maintenance-margin liquidation"})
            trades.to_csv(out/f"trades_{label}.csv",index=False)
            cycles.to_csv(out/f"cycles_{label}.csv",index=False)
            skips.to_csv(out/f"skips_{label}.csv",index=False)
            (out/f"summary_{label}.json").write_text(
                json.dumps(summary,indent=2,default=str),encoding="utf-8")
            rows.append(summary)
            print(f"[PERIOD COMPLETE] {run_no}/{total} {label} equity={summary.get('final_equity')} "
                f"pnl={summary.get('net_pnl')} dd={summary.get('max_drawdown_pct')} "
                f"trades={len(trades)} cycles={len(cycles)} "
                f"elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/"recovery_robustness_summary.csv",index=False)
    manifest={"oos_cutoff_utc":CUTOFF_UTC,"periods":PERIODS,
        "risk_caps_percent":list(RISK_CAPS),"runs":total,
        "period_ranges_utc":[{"start":_utc(a),"end_exclusive":_utc(b)} for a,b in ranges],
        "independent_periods":True,"one_global_open_position":True,
        "costs":{"fees":0.0,"slippage":0.0,"funding":0.0},"live_trading":False}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (out/"README.md").write_text(
        "# Serialized recovery — three-period OOS robustness validation\n\n"
        f"- OOS cutoff: {CUTOFF_UTC}\n"
        "- Same validated 76-symbol universe and prepared OOS signals.\n"
        "- Three adjacent equal-duration periods; each starts with identical initial capital and a flat portfolio.\n"
        "- Positions/recovery cycles cannot cross period boundaries; unresolved cycles are reported at period end.\n"
        "- Each period tests unlimited exact recovery to initial target-profit with 3% and 5% aggregate open-risk caps.\n"
        "- One open position globally; active symbol owns the lock for its complete recovery cycle.\n"
        "- BTC 1d+4h, ATR 2.0, RR 2.0, base risk 0.5%; fees/slippage/funding zero.\n"
        "- Equity wipeout is a conservative proxy, not exchange maintenance-margin liquidation.\n"
        "- Six independent historical OOS simulations; no live trading.\n",encoding="utf-8")
    print(df.to_string(index=False),flush=True)
    print(f"[COMPLETE] {len(rows)}/{total} simulations",flush=True)

if __name__=="__main__": run()
