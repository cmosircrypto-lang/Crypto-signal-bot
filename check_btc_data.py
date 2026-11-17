import pandas as pd
from datetime import datetime, timezone

files = [
    "data/BTC-SWAP-USDT_5m.parquet",
    "data/BTC-SWAP-USDT_15m.parquet",
    "data/BTC-SWAP-USDT_1h.parquet",
    "data/BTC-SWAP-USDT_4h.parquet",
    "data/BTC-SWAP-USDT_1d.parquet",
]

for file in files:
    print("\n" + "=" * 70)
    print(file)

    try:
        df = pd.read_parquet(file)

        print("Rows:", len(df))

        first_ts = int(df["timestamp"].iloc[0])
        last_ts = int(df["timestamp"].iloc[-1])

        first_date = datetime.fromtimestamp(
            first_ts / 1000,
            tz=timezone.utc
        )

        last_date = datetime.fromtimestamp(
            last_ts / 1000,
            tz=timezone.utc
        )

        print("Start:", first_date)
        print("End  :", last_date)

        days = (last_date - first_date).days
        print("Days :", days)

    except Exception as e:
        print("ERROR:", e)

print("\nDone.")
