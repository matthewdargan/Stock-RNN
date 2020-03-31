"""LSTM model used for stock prediction."""
import matplotlib.pyplot as plt
from pandas import DataFrame
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GRU,
    MaxPool1D,
)

from preprocess import preprocess


def plot_predictions(actual, predictions, save=False, fig_name=None):
    """Plot stock history and predictions to compare results."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual.iloc[:, 0], label="History")
    plt.plot(predictions, label="Prediction")
    plt.title("Training Predictions")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend(loc="upper left")

    if save:
        plt.savefig(f"figures/{fig_name}")

    plt.show()


def plot_losses(loss, val_loss, save=False, fig_name=None):
    """Plot training loss and validation loss."""
    plt.figure(figsize=(12, 6))
    plt.plot(loss, label="loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()

    if save:
        plt.savefig(f"figures/{fig_name}")

    plt.show()


BATCH_SIZE = 64
EPOCHS = 50

stock_data, stock_labels, stock_scaler, label_scaler = preprocess("data/KO.csv")
stock_time_series = stock_data.copy()

# Add dimension for timestep size
stock_data = stock_data.to_numpy().reshape(stock_data.shape[0], 1, stock_data.shape[1])

conv_model = Sequential(
    [
        Conv1D(5, kernel_size=1, activation="relu", input_shape=stock_data.shape[1:],),
        MaxPool1D(pool_size=1),
        GRU(128, return_sequences=True),
        Dropout(0.2),
        GRU(128, return_sequences=True),
        GRU(128),
        Dropout(0.2),
        Dense(28, activation="relu"),
        Dense(1),
    ]
)

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
        validation_split=0.3,
        shuffle=True,
    )

    stock_pred = label_scaler.inverse_transform(model.predict(stock_data))
    stock_time_series = DataFrame(stock_scaler.inverse_transform(stock_time_series))

    plot_predictions(stock_time_series, stock_pred)
    plot_losses(history.history["loss"], history.history["val_loss"])
