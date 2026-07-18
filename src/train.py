import os

from src.data_loader import DataLoader
from src.preprocess import TextPreprocessor
from src.model import SentimentLSTM


class ModelTrainer:
    def __init__(self):
        self.data_path = "data/IMDB_Dataset_CLEANED.csv"
        self.model_path = "models/sentiment_model.keras"
        self.tokenizer_path = "models/tokenizer.pkl"

        os.makedirs("models", exist_ok=True)

    def train(self):

        # -----------------------------
        # Load Dataset
        # -----------------------------
        print("=" * 50)
        print("Loading Dataset...")
        print("=" * 50)

        loader = DataLoader(self.data_path)

        loader.load_data()
        loader.preprocess_labels()

        X_train, X_test, y_train, y_test = loader.split_data()

        # -----------------------------
        # Preprocessing
        # -----------------------------
        print("\nCleaning Reviews...")

        preprocessor = TextPreprocessor(
            max_words=10000,
            max_length=200
        )

        X_train = preprocessor.clean_reviews(X_train)
        X_test = preprocessor.clean_reviews(X_test)

        print("Tokenizing...")

        preprocessor.fit_tokenizer(X_train)

        X_train = preprocessor.texts_to_sequences(X_train)
        X_test = preprocessor.texts_to_sequences(X_test)

        X_train = preprocessor.pad_sequences_data(X_train)
        X_test = preprocessor.pad_sequences_data(X_test)

        # Save tokenizer
        preprocessor.save_tokenizer(self.tokenizer_path)

        # -----------------------------
        # Build Model
        # -----------------------------
        print("\nBuilding LSTM Model...")

        builder = SentimentLSTM(
            vocab_size=10000,
            embedding_dim=128,
            max_length=200,
            lstm_units=128
        )

        model = builder.build_model()

        model.summary()

        # -----------------------------
        # Train Model
        # -----------------------------
        print("\nTraining Started...\n")

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=5,
            batch_size=64,
            verbose=1
        )

        # -----------------------------
        # Evaluate
        # -----------------------------
        print("\nEvaluating Model...\n")

        loss, accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=1
        )

        print(f"\nTest Accuracy : {accuracy:.4f}")
        print(f"Test Loss     : {loss:.4f}")

        # -----------------------------
        # Save Model
        # -----------------------------
        model.save(self.model_path)

        print("\nModel Saved Successfully!")
        print("Tokenizer Saved Successfully!")

        return history


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.train()