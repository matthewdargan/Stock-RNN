"""LSTM model used for stock prediction."""
import matplotlib.pyplot as plt
from pandas import DataFrame
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU

from preprocess import preprocess


def plot_predictions(actual, predictions):
    """Plot stock history and predictions to compare results."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual.iloc[:, 0], label="History")
    plt.plot(predictions, label="Prediction")
    plt.legend(loc="upper left")
    plt.show()


def plot_losses(loss, val_loss):
    """Plot training loss and validation loss."""
    plt.figure(figsize=(12, 6))
    plt.plot(loss, label="loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend()
    plt.show()


BATCH_SIZE = 64
EPOCHS = 50

stock_data, stock_labels, stock_scaler, label_scaler = preprocess("data/GE.csv")
stock_time_series = stock_data.copy()

# Add dimension for timestep size
stock_data = stock_data.to_numpy().reshape(stock_data.shape[0], 1, stock_data.shape[1])

model = Sequential(
    [
        GRU(128, return_sequences=True, input_shape=stock_data.shape[1:]),
        Dropout(0.2),
        GRU(128, return_sequences=True),
        GRU(128),
        Dropout(0.2),
        Dense(28, activation="relu"),
        Dense(1),
    ]
)
model.compile(loss="mse", optimizer="adamax", metrics=["mae", "mse"])

if __name__ == "__main__":
    history = model.fit(
        stock_data,
        stock_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.4,
        shuffle=True,
    )

    stock_pred = label_scaler.inverse_transform(model.predict(stock_data))
    stock_time_series = DataFrame(stock_scaler.inverse_transform(stock_time_series))

    plot_predictions(stock_time_series, stock_pred)
    plot_losses(history.history["loss"], history.history["val_loss"])
