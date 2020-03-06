"""LSTM model used for stock prediction."""
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

EPOCHS = 50

model = Sequential(
    [
        LSTM(10, input_shape=(23,)),
        Dense(128, activation="gelu"),
        Dense(64, activation="gelu"),
        Dense(10, activation="gelu"),
        Dense(2, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

if __name__ == "__main__":
    stock_data, next_day_close_labels, scaler = preprocess("./data/MCD.csv")
    stock_train, stock_test, label_train, label_test = train_test_split(
        stock_data, next_day_close_labels, test_size=0.3, shuffle=False
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (stock_train.values, label_train)
    ).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (stock_test.values, label_test)
    ).batch(32)

    model.fit(
        train_dataset, epochs=EPOCHS, validation_data=test_dataset,
    )

    # Denormalize data back to original scale
    # scaler.inverse_transform(stock_data)
