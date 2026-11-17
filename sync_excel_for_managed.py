import json
import os
import sys

# ایمپورت ماژول اکسل
sys.path.insert(0, os.getcwd())
try:
    import excel_log
except ImportError:
    print("❌ ماژول excel_log پیدا نشد. مطمئن شوید main.py و excel_log.py در همین پوشه هستند.")
    sys.exit(1)

FILES = [
    ("last_signals_v0a.json", "crypto-signal-bot-v0a"),
    ("last_signals_v60a.json", "crypto-signal-bot-v60a"),
    ("last_signals_v70a.json", "crypto-signal-bot-v70a"),
]

def sync_excel():
    for json_file, bot_name in FILES:
        if not os.path.exists(json_file):
            print(f"⏩ {json_file} وجود ندارد.")
            continue
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        open_trades = {k: v for k, v in data.items() 
                       if not k.startswith("__") and v.get("status") == "OPEN"}
        
        if not open_trades:
            print(f"⏩ {json_file}: معامله باز ندارد.")
            continue
        
        for symbol, trade in open_trades.items():
            entry = float(trade["entry"])
            tp = float(trade["tp"])
            sl = float(trade["sl"])
            direction = trade["direction"]
            
            # ثبت در اکسل
            excel_log.log_signal(bot_name, symbol, direction, entry, tp, sl)
            print(f"✅ {symbol} در اکسل {bot_name} ثبت شد.")

if __name__ == "__main__":
    print("🚀 شروع همگام‌سازی اکسل با معاملات باز...\n")
    sync_excel()
    print("🎉 تمام شد.")