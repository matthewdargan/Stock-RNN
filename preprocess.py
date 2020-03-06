"""Preprocess stock data."""
import numpy as np
from pandas import DataFrame, read_csv, to_datetime
from sklearn.preprocessing import MinMaxScaler
from ta.trend import CCIIndicator, ema_indicator, macd
from ta.momentum import roc, rsi, wr
from ta.volatility import average_true_range, bollinger_mavg
from typing import Tuple


def preprocess(csv_path: str) -> Tuple[DataFrame, MinMaxScaler]:
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

    compute_indicators(df)

    # Add indices to include information of the larger market movement
    add_spx(df)
    add_vix(df)

    df.dropna(inplace=True)

    # Normalize datapoints between 0 and 1
    scaler = normalize(df)

    # Convert timestamp column to timestamp datatype and set it the as index
    df["timestamp"] = to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    return df, scaler


def compute_indicators(df: DataFrame) -> DataFrame:
    """Compute useful stock market indicators for the dataframe."""
    df["rsi14"] = rsi(df["close"])
    df["macd"] = macd(df["close"])
    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"]).cci()
    df["atr"] = average_true_range(df["high"], df["low"], df["close"])
    df["boll"] = bollinger_mavg(df["close"])
    df["roc"] = roc(df["close"])
    df["wpr"] = wr(df["high"], df["low"], df["close"])
    df["ema50"] = ema_indicator(df["close"], 50)
    df["ema100"] = ema_indicator(df["close"], 100)
    df["ema200"] = ema_indicator(df["close"], 200)


def normalize(df: DataFrame) -> MinMaxScaler:
    """Normalize dataframe to be used inside the neural network."""
    columns = [
        "close",
        "volume",
        "open",
        "high",
        "low",
        "rsi14",
        "macd",
        "cci",
        "atr",
        "boll",
        "roc",
        "wpr",
        "ema50",
        "ema100",
        "ema200",
        "spx_open",
        "spx_high",
        "spx_low",
        "spx_close",
        "vix_open",
        "vix_high",
        "vix_low",
        "vix_close",
    ]
    values = df[columns].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = scaler.fit_transform(values)
    df[columns] = values

    return scaler


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


if __name__ == "__main__":
    preprocess("./data/MCD.csv")
