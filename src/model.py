from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


class SentimentLSTM:
    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=128,
        max_length=200,
        lstm_units=128,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units

    def build_model(self):
        """
        Build and compile the LSTM model.
        """

        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length
            ),

            LSTM(
                self.lstm_units,
                return_sequences=False
            ),

            Dropout(0.5),

            Dense(64, activation="relu"),

            Dropout(0.3),

            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model


# -------------------------------------------------
# Test Model
# -------------------------------------------------
if __name__ == "__main__":

    lstm = SentimentLSTM(
        vocab_size=10000,
        embedding_dim=128,
        max_length=200,
        lstm_units=128
    )

    model = lstm.build_model()

    model.summary()