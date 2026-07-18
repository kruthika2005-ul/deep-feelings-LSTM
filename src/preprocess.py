import re
import string
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextPreprocessor:
    def __init__(self, max_words=10000, max_length=200):
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

    def clean_text(self, text):
        """
        Clean review text.
        """
        if not isinstance(text, str):
            text = str(text)

        # Lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_reviews(self, reviews):
        """
        Clean a list/Series of reviews.
        """
        return reviews.apply(self.clean_text)

    def fit_tokenizer(self, reviews):
        """
        Fit tokenizer on training reviews.
        """
        self.tokenizer.fit_on_texts(reviews)

    def texts_to_sequences(self, reviews):
        """
        Convert text to sequences.
        """
        return self.tokenizer.texts_to_sequences(reviews)

    def pad_sequences_data(self, sequences):
        """
        Pad sequences to equal length.
        """
        return pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding="post",
            truncating="post"
        )

    def save_tokenizer(self, path):
        """
        Save tokenizer.
        """
        with open(path, "wb") as file:
            pickle.dump(self.tokenizer, file)

    def load_tokenizer(self, path):
        """
        Load tokenizer.
        """
        with open(path, "rb") as file:
            self.tokenizer = pickle.load(file)


# ------------------------------------------------------
# Test Preprocessor
# ------------------------------------------------------
if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv("data/IMDB_Dataset_CLEANED.csv")

    preprocessor = TextPreprocessor()

    # Clean reviews
    reviews = preprocessor.clean_reviews(df["review"])

    print("Original Review:\n")
    print(df["review"][0])

    print("\nCleaned Review:\n")
    print(reviews[0])

    # Tokenizer
    preprocessor.fit_tokenizer(reviews)

    sequences = preprocessor.texts_to_sequences(reviews)

    padded = preprocessor.pad_sequences_data(sequences)

    print("\nVocabulary Size:", len(preprocessor.tokenizer.word_index))
    print("Padded Shape:", padded.shape)

    # Save tokenizer
    preprocessor.save_tokenizer("models/tokenizer.pkl")

    print("\nTokenizer saved successfully.")