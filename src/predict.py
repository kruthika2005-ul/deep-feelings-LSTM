import numpy as np
from tensorflow.keras.models import load_model

from src.preprocess import TextPreprocessor


class SentimentPredictor:

    def __init__(
        self,
        model_path="models/sentiment_model.keras",
        tokenizer_path="models/tokenizer.pkl",
        max_length=200,
    ):

        self.model = load_model(model_path)

        self.preprocessor = TextPreprocessor(
            max_words=10000,
            max_length=max_length
        )

        self.preprocessor.load_tokenizer(tokenizer_path)

    def predict(self, review):

        # Clean review
        review = self.preprocessor.clean_text(review)

        # Convert to sequence
        sequence = self.preprocessor.texts_to_sequences([review])

        # Pad sequence
        sequence = self.preprocessor.pad_sequences_data(sequence)

        # Prediction
        probability = self.model.predict(sequence, verbose=0)[0][0]

        if probability >= 0.5:
            sentiment = "Positive 😊"
            confidence = probability
        else:
            sentiment = "Negative 😞"
            confidence = 1 - probability

        return sentiment, float(confidence)


# ---------------------------------------------------
# Test Prediction
# ---------------------------------------------------
if __name__ == "__main__":

    predictor = SentimentPredictor()

    while True:

        review = input("\nEnter a movie review (type 'exit' to quit): ")

        if review.lower() == "exit":
            break

        sentiment, confidence = predictor.predict(review)

        print(f"\nPrediction : {sentiment}")
        print(f"Confidence : {confidence:.2%}")