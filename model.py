"""LSTM model used for stock prediction."""
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from preprocess import preprocess


def plot_predictions(history, predictions):
    """Plot stock history and predictions to compare results."""
    plt.figure(figsize=(12, 6))
    plt.plot(history.iloc[:, 0], label="History")
    plt.plot(predictions, label="Prediction")
    plt.legend(loc="upper left")
    plt.show()


BATCH_SIZE = 64
EPOCHS = 50

stock_data, next_day_close_labels, stock_scaler, label_scaler = preprocess(
    "./data/MCD.csv"
)
stock_time_series = stock_data.copy()
stock_data = stock_data.to_numpy().reshape(stock_data.shape[0], 1, stock_data.shape[1])
stock_train, stock_test, label_train, label_test = train_test_split(
    stock_data, next_day_close_labels, test_size=0.3, shuffle=False
)

model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=stock_data.shape[1:]),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ]
)
model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])

if __name__ == "__main__":
    model.fit(
        stock_train,
        label_train,
        epochs=EPOCHS,
        validation_data=(stock_test, label_test),
    )

    stock_pred = label_scaler.inverse_transform(model.predict(stock_data))
    stock_time_series = DataFrame(stock_scaler.inverse_transform(stock_time_series))

    plot_predictions(stock_time_series, stock_pred)
