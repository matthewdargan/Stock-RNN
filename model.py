"""LSTM model used for stock prediction."""
from pandas import DataFrame
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

BATCH_SIZE = 64
EPOCHS = 50

stock_data, next_day_close_labels, stock_scaler, label_scaler = preprocess(
    "./data/MCD.csv"
)
hahahaha = stock_data.copy()
stock_data = stock_data.to_numpy().reshape(stock_data.shape[0], 1, stock_data.shape[1])
stock_train, stock_test, label_train, label_test = train_test_split(
    stock_data, next_day_close_labels, test_size=0.3, shuffle=False
)

# train_dataset = (
#     tf.data.Dataset.from_tensor_slices((stock_train, label_train))
#     # .shuffle(len(stock_train))
#     .batch(BATCH_SIZE).repeat()
# )
# test_dataset = (
#     tf.data.Dataset.from_tensor_slices((stock_test, label_test))
#     # .shuffle(len(stock_test))
#     .batch(BATCH_SIZE).repeat()
# )

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
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

if __name__ == "__main__":
    model.fit(
        stock_train,
        label_train,
        epochs=EPOCHS,
        # steps_per_epoch=len(stock_train),
        validation_data=(stock_test, label_test),
        # validation_steps=len(stock_test),
    )

    # TODO look into which columns are which, might be why mae is so high
    pred = model.predict(stock_test)
    pred = label_scaler.inverse_transform(pred)
    hahahaha = DataFrame(stock_scaler.inverse_transform(hahahaha))
    print(f"test: {hahahaha}, pred: {pred}")

    # Denormalize data back to original scale
    # scaler.inverse_transform(stock_data)
