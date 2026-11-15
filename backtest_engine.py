def mock_get_data(ctx, symbol, timeframe):
    df_full = load_df_with_indicators(symbol, timeframe)
    
    # همگام‌سازی تایم‌فریم: کندل‌های بالاتر باید حتماً بسته شده باشند
    if timeframe == "4h":
        # آخرین کندل 4 ساعته‌ای که کامل بسته شده
        current_ts_rounded = (CURRENT_TS // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
    elif timeframe == "1d":
        current_ts_rounded = (CURRENT_TS // (24 * 60 * 60 * 1000)) * (24 * 60 * 60 * 1000)
    elif timeframe == "15m":
        current_ts_rounded = (CURRENT_TS // (15 * 60 * 1000)) * (15 * 60 * 1000)
    else:
        current_ts_rounded = CURRENT_TS
    
    df_slice = df_full[df_full["timestamp"] <= current_ts_rounded].copy()
    
    if len(df_slice) < 40:
        raise ValueError(f"Not enough data at {CURRENT_TS}")
    return df_slice