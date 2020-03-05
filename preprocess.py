"""Preprocess stock data."""
from pandas import DataFrame, read_csv
from ta import average_true_range, bollinger_mavg, cci, macd, ema, roc, rsi, wr


def preprocess(csv: str) -> DataFrame:
    """Preprocess stock market data to prepare it to be used as training and testing data for the model."""
    df = read_csv(csv, sep=", ", engine="python")
    df.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close/Last": "Close"}, inplace=True
    )

    # Remove dollar signs from entries
    df["Open"] = df["Open"].str.replace("$", "").astype(float)
    df["High"] = df["High"].str.replace("$", "").astype(float)
    df["Low"] = df["Low"].str.replace("$", "").astype(float)
    df["Close"] = df["Close"].str.replace("$", "").astype(float)

    df = compute_indicators(df)

    # TODO add S&P 500 and CBOE Volitity indices as a column

    # df = normalize(df)

    return df


def compute_indicators(df: DataFrame) -> DataFrame:
    """Compute useful stock market indicators for the dataframe."""
    df["rsi14"] = rsi(df["close"])
    df["macd"] = macd(df["close"])
    df["cci"] = cci(df["high"], df["low"], df["close"])
    df["atr"] = average_true_range(df["high"], df["low"], df["close"])
    df["bool"] = bollinger_mavg(df["close"])
    df["roc"] = roc(df["close"])
    df["wpr"] = wr(df["high"], df["low"], df["close"])
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], 200)

    return df


def normalize(df: DataFrame) -> DataFrame:
    """Normalize dataframe to be used inside the neural network."""
    # TODO normalize dataframe
    pass


if __name__ == "__main__":
    preprocess("./data/MCD.csv")
