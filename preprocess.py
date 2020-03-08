"""Preprocess stock data."""
import numpy as np
from pandas import DataFrame, read_csv, Series, to_datetime
from sklearn.preprocessing import MinMaxScaler
from ta.trend import CCIIndicator, ema_indicator, macd
from ta.momentum import roc, rsi, wr
from ta.volatility import average_true_range, bollinger_mavg
from typing import Tuple


def preprocess(csv_path: str) -> Tuple[DataFrame, Series, MinMaxScaler, MinMaxScaler]:
    """
    Preprocess stock market data to prepare it to be used as training and testing
    data for the model.
    """
    df = read_csv(csv_path, sep=", ", engine="python")
    df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close/Last": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # Remove dollar signs from entries
    df["open"] = df["open"].str.replace("$", "").astype(np.float64)
    df["high"] = df["high"].str.replace("$", "").astype(np.float64)
    df["low"] = df["low"].str.replace("$", "").astype(np.float64)
    df["close"] = df["close"].str.replace("$", "").astype(np.float64)

    # compute_indicators(df)

    # Add indices to include information of the larger market movement
    add_spx(df)
    add_vix(df)

    # Convert timestamp column to timestamp datatype and set it the as index
    df["timestamp"] = to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    df.dropna(inplace=True)

    # Get labels for each row in the dataframe
    labels = get_labels(df)

    # Normalize datapoints between 0 and 1
    df, df_scaler = normalize_dataframe(df)
    labels, label_scaler = normalize_labels(labels)

    return df, labels, df_scaler, label_scaler


def compute_indicators(df: DataFrame) -> DataFrame:
    """Compute useful stock market indicators for the dataframe."""
    df["rsi14"] = rsi(df["close"])
    df["macd"] = macd(df["close"])
    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"]).cci()
    df["atr"] = average_true_range(df["high"], df["low"], df["close"])
    df["boll"] = bollinger_mavg(df["close"])
    df["roc"] = roc(df["close"])
    df["wpr"] = wr(df["high"], df["low"], df["close"])
    # df["ema50"] = ema_indicator(df["close"], 50)
    # df["ema100"] = ema_indicator(df["close"], 100)
    # df["ema200"] = ema_indicator(df["close"], 200)


def add_spx(df: DataFrame) -> None:
    """Add S&P 500 index to dataframe."""
    spx_data = read_csv("./data/SPX.csv", sep=", ", engine="python")
    df["spx_open"] = spx_data["Open"].astype(np.float64)
    df["spx_high"] = spx_data["High"].astype(np.float64)
    df["spx_low"] = spx_data["Low"].astype(np.float64)
    df["spx_close"] = spx_data["Close/Last"].astype(np.float64)


def add_vix(df: DataFrame) -> None:
    """Add CBOE Volatility Index to dataframe."""
    vix_data = read_csv("./data/VIX.csv")[1553:].reset_index()
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


def get_labels(df: DataFrame) -> Series:
    """Get labels for a dataframe (closing values from the next day)."""
    closing_values = df["close"].values
    closing_values = np.delete(closing_values, 0)

    # Remove last row from original dataframe since we do not have a label for it
    df.drop(df.tail(1).index, inplace=True)

    return closing_values


def normalize_labels(labels: Series) -> Tuple[Series, MinMaxScaler]:
    """Normalize label values."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    labels = scaler.fit_transform(labels.reshape(-1, 1))
    return labels, scaler


if __name__ == "__main__":
    preprocess("./data/MCD.csv")
