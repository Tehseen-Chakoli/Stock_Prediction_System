from pathlib import Path

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_csv(filename: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / filename)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_price_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df[price_col].pct_change(1)
    df["return_3d"] = df[price_col].pct_change(3)
    df["return_7d"] = df[price_col].pct_change(7)

    df["ma_10"] = df[price_col].rolling(window=10).mean()
    df["ma_20"] = df[price_col].rolling(window=20).mean()
    df["ma_50"] = df[price_col].rolling(window=50).mean()

    df["volatility_10"] = df["return_1d"].rolling(window=10).std()
    df["volatility_20"] = df["return_1d"].rolling(window=20).std()

    df["rsi_14"] = RSIIndicator(close=df[price_col], window=14).rsi()

    macd = MACD(close=df[price_col], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    return df


def add_market_context_features(
    sbi_df: pd.DataFrame, nifty_df: pd.DataFrame, banknifty_df: pd.DataFrame
) -> pd.DataFrame:
    nifty = nifty_df[["Date", "Close"]].copy()
    nifty.rename(columns={"Close": "nifty_close"}, inplace=True)
    nifty["nifty_return_1d"] = nifty["nifty_close"].pct_change(1)
    nifty["nifty_return_3d"] = nifty["nifty_close"].pct_change(3)
    nifty["nifty_return_7d"] = nifty["nifty_close"].pct_change(7)

    banknifty = banknifty_df[["Date", "Close"]].copy()
    banknifty.rename(columns={"Close": "banknifty_close"}, inplace=True)
    banknifty["banknifty_return_1d"] = banknifty["banknifty_close"].pct_change(1)
    banknifty["banknifty_return_3d"] = banknifty["banknifty_close"].pct_change(3)
    banknifty["banknifty_return_7d"] = banknifty["banknifty_close"].pct_change(7)

    merged = sbi_df.merge(nifty, on="Date", how="left")
    merged = merged.merge(banknifty, on="Date", how="left")

    merged["relative_return_vs_nifty"] = merged["return_1d"] - merged["nifty_return_1d"]
    merged["relative_return_vs_banknifty"] = merged["return_1d"] - merged["banknifty_return_1d"]

    return merged


def add_targets(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    df = df.copy()

    df["future_close_7d"] = df[price_col].shift(-7)
    df["future_return_7d"] = (df["future_close_7d"] - df[price_col]) / df[price_col]
    df["target_up_7d"] = (df["future_return_7d"] > 0).astype(int)

    return df


def build_feature_table() -> pd.DataFrame:
    sbi = load_csv("SBIN.NS.csv")
    nifty = load_csv("NSEI.csv")
    banknifty = load_csv("NSEBANK.csv")

    sbi = add_price_features(sbi, price_col="Close")
    feature_df = add_market_context_features(sbi, nifty, banknifty)
    feature_df = add_targets(feature_df, price_col="Close")

    feature_df = feature_df.sort_values("Date").reset_index(drop=True)

    return feature_df


def save_feature_table(df: pd.DataFrame, filename: str = "sbi_features.csv") -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved feature table at {output_path}")


def run() -> None:
    df = build_feature_table()
    save_feature_table(df)

    print("\nFeature table preview:")
    print(df.head(15))

    print("\nShape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    run()