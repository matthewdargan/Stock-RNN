"""LSTM model used for stock prediction."""
from tensorflow.keras.layers import Dense, LSTM, Sequential
from preprocess import preprocess

EPOCHS = 50

model = Sequential(
    [
        LSTM(10, input_shape=(10,)),
        Dense(128, activation="gelu"),
        Dense(64, activation="gelu"),
        Dense(10, activation="gelu"),
        Dense(2, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(stock_data, epochs=EPOCHS, validation_data=stock_valid_data)

if __name__ == "__main__":
    # Preprocess data
    data, scaler = preprocess("./data/MCD.csv")

    # Denormalize data back to original scale
    scaler.inverse_transform(data)
