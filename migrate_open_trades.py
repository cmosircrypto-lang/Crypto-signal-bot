"""
انتقال معاملات باز از ربات‌های اصلی به ربات‌های مدیریت سرمایه (v0a, v60a, v70a)
"""
import json
import os
import copy

PAIRS = [
    ("last_signals.json", "last_signals_v0a.json"),
    ("last_signals_v60.json", "last_signals_v60a.json"),
    ("last_signals_v70.json", "last_signals_v70a.json"),
]

INITIAL_BALANCE = 100.0

def load(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def migrate():
    for src_file, dst_file in PAIRS:
        src = load(src_file)
        dst = load(dst_file)
        open_trades = {k: v for k, v in src.items()
                       if not k.startswith("__") and v.get("status") == "OPEN"}
        if not open_trades:
            print(f"⏩ {src_file}: معامله باز ندارد، رد شد.")
            continue
        if "__balance__" not in dst:
            dst["__balance__"] = INITIAL_BALANCE
        if "__in_recovery__" not in dst:
            dst["__in_recovery__"] = False
        if "__recovery_losses__" not in dst:
            dst["__recovery_losses__"] = []
        balance = dst["__balance__"]
        for symbol, trade in open_trades.items():
            new_trade = copy.deepcopy(trade)
            entry = float(trade["entry"])
            sl = float(trade["sl"])
            tp = float(trade["tp"])
            sl_dist_pct = abs(entry - sl) / entry if entry > 0 else 0
            risk_amount = balance * 0.03
            position_value = risk_amount / sl_dist_pct if sl_dist_pct > 0 else 0
            leverage = position_value / balance if balance > 0 else 1.0
            position_size = position_value / entry if entry > 0 else 0
            new_trade["position_size"] = round(position_size, 6)
            new_trade["position_value"] = round(position_value, 2)
            new_trade["leverage"] = round(max(1.0, leverage), 2)
            new_trade["risk_amount"] = round(risk_amount, 2)
            new_trade["migrated_from"] = src_file
            dst[symbol] = new_trade
            print(f"✅ کپی شد: {symbol} از {src_file} به {dst_file}")
        save(dst_file, dst)
        print(f"💾 {dst_file} ذخیره شد ({len(open_trades)} معامله).\n")

if __name__ == "__main__":
    print("🚀 شروع انتقال معاملات باز...\n")
    migrate()
    print("🎉 تمام شد.")