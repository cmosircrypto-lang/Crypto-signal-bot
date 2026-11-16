import os
import pandas as pd
from datetime import datetime, timezone

DATA_DIR = "data"

def check_data_integrity():
    if not os.path.exists(DATA_DIR):
        print("❌ پوشه data وجود ندارد!")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith("_1d.parquet")]
    
    if not files:
        print("❌ هیچ فایل 1d در پوشه data پیدا نشد!")
        return

    print(f"🔍 بررسی {len(files)} فایل در پوشه data...\n")
    
    all_good = True
    for f in sorted(files):
        path = os.path.join(DATA_DIR, f)
        try:
            df = pd.read_parquet(path)
            count = len(df)
            start_ts = int(df["timestamp"].iloc[0]) / 1000
            end_ts = int(df["timestamp"].iloc[-1]) / 1000
            
            start_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime('%Y-%m-%d')
            
            # ۳۶۵ روز یعنی حداقل ۳۶۰ کندل (با احتساب کمی اختلاف)
            if count < 360:
                status = f"⚠️ کمبود داده ({count} کندل)"
                all_good = False
            else:
                status = f"✅ {count} کندل"

            print(f"{f:30} | {status:20} | از {start_date} تا {end_date}")
            
        except Exception as e:
            print(f"❌ {f}: خطا در خواندن ({e})")
            all_good = False

    print("\n" + "="*50)
    if all_good:
        print("🎉 همه فایل‌ها ۳۶۵ روز داده معتبر دارند!")
    else:
        print("⚠️ برخی فایل‌ها داده کافی ندارند. لطفاً دوباره دانلود کنید.")

if __name__ == "__main__":
    check_data_integrity()