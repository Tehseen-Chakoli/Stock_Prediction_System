import os
from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("data/raw")


def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker from Yahoo Finance.
    Returns a clean flat dataframe.
    """
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise ValueError(f"No data fetched for ticker: {ticker}")

    # Flatten columns if yfinance returns a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()

    # Keep only expected columns if present
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    existing_cols = [col for col in expected_cols if col in df.columns]
    df = df[existing_cols].copy()

    # Add ticker column
    df["Ticker"] = ticker

    # Clean types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def save_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Save dataframe to CSV.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{ticker.replace('^', '')}.csv"
    filepath = RAW_DATA_DIR / filename

    df.to_csv(filepath, index=False)
    print(f"Saved data for {ticker} at {filepath}")


def run() -> None:
    tickers = ["SBIN.NS", "^NSEI", "^NSEBANK"]

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = fetch_stock_data(ticker=ticker)
        save_data(df=df, ticker=ticker)


if __name__ == "__main__":
    run()