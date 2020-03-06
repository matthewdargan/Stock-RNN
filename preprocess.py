"""Preprocess stock data."""
from pandas import DataFrame, read_csv
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
        },
        inplace=True,
    )

    # Remove dollar signs from entries
    df["open"] = df["open"].str.replace("$", "").astype(float)
    df["high"] = df["high"].str.replace("$", "").astype(float)
    df["low"] = df["low"].str.replace("$", "").astype(float)
    df["close"] = df["close"].str.replace("$", "").astype(float)

    compute_indicators(df)

    # Add indices for representing movement of the overall market
    add_spx(df)
    add_vix(df)

    scaler = normalize(df)

    print(df)
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
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(df.values)
    df = DataFrame(values_scaled)
    return scaler


def add_spx(df: DataFrame) -> None:
    """Add S&P 500 index to dataframe."""
    spx_data = read_csv("./data/SPX.csv", sep=", ", engine="python")
    df["spx_open"] = spx_data["Open"].astype(float)
    df["spx_high"] = spx_data["High"].astype(float)
    df["spx_low"] = spx_data["Low"].astype(float)
    df["spx_close"] = spx_data["Close/Last"].astype(float)


def add_vix(df: DataFrame) -> None:
    """Add CBOE Volatility Index to dataframe."""
    # TODO fix NAs
    vix_data = read_csv("./data/VIX.csv")[1553:]
    df["vix_open"] = vix_data["Open"].astype(float)
    df["vix_high"] = vix_data["High"].astype(float)
    df["vix_low"] = vix_data["Low"].astype(float)
    df["vix_close"] = vix_data["Close"].astype(float)


if __name__ == "__main__":
    preprocess("./data/MCD.csv")
