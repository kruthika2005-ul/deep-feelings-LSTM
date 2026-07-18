import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from tensorflow.keras.models import load_model

from src.data_loader import DataLoader
from src.preprocess import TextPreprocessor


class ModelEvaluator:

    def __init__(
        self,
        data_path="data/IMDB_Dataset_CLEANED.csv",
        model_path="models/sentiment_model.keras",
        tokenizer_path="models/tokenizer.pkl",
    ):

        self.data_path = data_path

        self.model = load_model(model_path)

        self.preprocessor = TextPreprocessor(
            max_words=10000,
            max_length=200,
        )

        self.preprocessor.load_tokenizer(tokenizer_path)

    def evaluate(self):

        # -----------------------------
        # Load Dataset
        # -----------------------------
        loader = DataLoader(self.data_path)

        loader.load_data()
        loader.preprocess_labels()

        _, X_test, _, y_test = loader.split_data()

        # -----------------------------
        # Preprocess
        # -----------------------------
        X_test = self.preprocessor.clean_reviews(X_test)

        X_test = self.preprocessor.texts_to_sequences(X_test)

        X_test = self.preprocessor.pad_sequences_data(X_test)

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = self.model.predict(X_test, verbose=0)

        y_pred = (predictions >= 0.5).astype(int)

        # -----------------------------
        # Metrics
        # -----------------------------
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n========== MODEL PERFORMANCE ==========\n")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report\n")
        print(classification_report(y_test, y_pred))

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Negative", "Positive"]
        )

        disp.plot(cmap="Blues")

        plt.title("Confusion Matrix")
        plt.show()


if __name__ == "__main__":

    evaluator = ModelEvaluator()

    evaluator.evaluate()