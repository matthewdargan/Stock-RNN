"""LSTM model used for stock prediction."""
from preprocess import preprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Sequential

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

if __name__ == "__main__":
    stock_data, scaler = preprocess("./data/MCD.csv")
    train, test = train_test_split(stock_data, test_size=0.3)

    model.fit(train, epochs=EPOCHS, validation_data=test)

    # Denormalize data back to original scale
    scaler.inverse_transform(stock_data)
