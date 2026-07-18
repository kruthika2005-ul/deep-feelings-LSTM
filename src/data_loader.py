import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the dataset."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def show_info(self):
        """Display basic dataset information."""
        print("\n========== Dataset Information ==========")
        print("Shape :", self.data.shape)
        print("\nColumns:")
        print(self.data.columns.tolist())

        print("\nFirst 5 Rows:")
        print(self.data.head())

        print("\nMissing Values:")
        print(self.data.isnull().sum())

        print("\nSentiment Distribution:")
        print(self.data["sentiment"].value_counts())

    def preprocess_labels(self):
        """Convert sentiment labels into numeric values."""
        self.data["sentiment"] = self.data["sentiment"].map({
            "positive": 1,
            "negative": 0
        })

        return self.data

    def split_data(self, test_size=0.2, random_state=42):
        """Split the dataset into training and testing sets."""

        X = self.data["review"]
        y = self.data["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# Test the DataLoader
# ---------------------------------------------------
if __name__ == "__main__":

    loader = DataLoader("data/IMDB_Dataset_CLEANED.csv")
    # Load dataset
    loader.load_data()

    # Show information
    loader.show_info()

    # Convert labels
    loader.preprocess_labels()

    # Split data
    X_train, X_test, y_train, y_test = loader.split_data()

    print("\n========== Data Split ==========")
    print("Training Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    print("\nSample Review:")
    print(X_train.iloc[0])

    print("\nEncoded Label:")
    print(y_train.iloc[0])