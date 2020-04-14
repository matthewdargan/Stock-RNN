"""Preprocess stock data."""
from typing import Tuple

import numpy as np
import talib
from pandas import DataFrame, read_csv, to_datetime
from sklearn.preprocessing import MinMaxScaler


def preprocess(
    csv_path: str,
) -> Tuple[DataFrame, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Preprocess stock market data to prepare it to be used as training and testing
    data for the model.
    """
    df = read_csv(csv_path)
    df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj close",
            "Volume": "volume",
        },
        inplace=True,
    )
    df["open"] = df["open"].astype(np.float64)
    df["high"] = df["high"].astype(np.float64)
    df["low"] = df["low"].astype(np.float64)
    df["close"] = df["close"].astype(np.float64)
    df["adj close"] = df["adj close"].astype(np.float64)
    df["volume"] = df["volume"].astype(np.uint64)

    # Convert timestamp column to timestamp datatype and set it as index
    df["timestamp"] = to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    compute_indicators(df)
    add_spx(df)
    add_vix(df)
    df.dropna(inplace=True)
    labels = get_daily_labels(df)

    # Normalize data points between 0 and 1
    df, df_scaler = normalize_dataframe(df)
    labels, label_scaler = normalize_labels(labels)

    return df, labels, df_scaler, label_scaler


def compute_indicators(df: DataFrame) -> None:
    """Compute useful stock market indicators for the dataframe."""
    df["ma5"] = talib.SMA(df["close"], timeperiod=5)
    df["ma10"] = talib.SMA(df["close"], timeperiod=10)
    df["ma50"] = talib.SMA(df["close"], timeperiod=50)
    df["ma200"] = talib.SMA(df["close"], timeperiod=200)
    df["rsi"] = talib.RSI(df["close"])
    df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(df["close"])
    df["cci"] = talib.CCI(df["high"], df["low"], df["close"])
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"])
    df["bbands_high"], df["bbands_middle"], df["bbands_low"] = talib.BBANDS(df["close"])
    df["roc"] = talib.ROC(df["close"])
    df["w%r"] = talib.WILLR(df["high"], df["low"], df["close"])
    df["ema50"] = talib.EMA(df["close"], timeperiod=50)
    df["ema100"] = talib.EMA(df["close"], timeperiod=100)
    df["ema200"] = talib.EMA(df["close"], timeperiod=200)


def add_spx(df: DataFrame) -> None:
    """Add S&P 500 index to dataframe."""
    spx_data = read_csv("data/^GSPC.csv")
    spx_data.rename(
        columns={"Date": "timestamp"}, inplace=True,
    )
    spx_data["timestamp"] = to_datetime(spx_data["timestamp"])
    spx_data.set_index("timestamp", inplace=True)

    df["spx_open"] = spx_data["Open"].astype(np.float64)
    df["spx_high"] = spx_data["High"].astype(np.float64)
    df["spx_low"] = spx_data["Low"].astype(np.float64)
    df["spx_close"] = spx_data["Close"].astype(np.float64)
    df["spx_volume"] = spx_data["Volume"].astype(np.uint64)


def add_vix(df: DataFrame) -> None:
    """Add CBOE Volatility Index to dataframe."""
    vix_data = read_csv("./data/^VIX.csv")
    vix_data.rename(
        columns={"Date": "timestamp"}, inplace=True,
    )
    vix_data["timestamp"] = to_datetime(vix_data["timestamp"])
    vix_data.set_index("timestamp", inplace=True)

    df["vix_open"] = vix_data["Open"].astype(np.float64)
    df["vix_high"] = vix_data["High"].astype(np.float64)
    df["vix_low"] = vix_data["Low"].astype(np.float64)
    df["vix_close"] = vix_data["Close"].astype(np.float64)


def normalize_dataframe(df: DataFrame) -> Tuple[DataFrame, MinMaxScaler]:
    """Normalize dataframe values."""
    values = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = scaler.fit_transform(values)
    df = DataFrame(values)

    return df, scaler


def get_daily_labels(df: DataFrame) -> np.ndarray:
    """Get labels for a dataframe (closing values from the next day)."""
    closing_values = df["close"].values
    closing_values = np.delete(closing_values, 0)

    # Remove last row from original dataframe since we do not have a label for it
    df.drop(df.tail(1).index, inplace=True)

    return closing_values


def get_weekly_labels(df: DataFrame) -> np.ndarray:
    """Get labels for a dataframe (closing values starting one week ahead)."""
    closing_values = df["close"].values
    closing_values = np.delete(closing_values, range(8))

    # Remove last row from original dataframe since we do not have a label for it
    df.drop(df.tail(8).index, inplace=True)

    return closing_values


def get_monthly_labels(df: DataFrame) -> np.ndarray:
    """Get labels for a dataframe (closing values starting one month ahead)."""
    closing_values = df["close"].values
    closing_values = np.delete(closing_values, range(31))

    # Remove last row from original dataframe since we do not have a label for it
    df.drop(df.tail(31).index, inplace=True)

    return closing_values


def normalize_labels(labels: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Normalize label values."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    labels = scaler.fit_transform(labels.reshape(-1, 1))
    return labels, scaler


if __name__ == "__main__":
    preprocess("data/KO.csv")
