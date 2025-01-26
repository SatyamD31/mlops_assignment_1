import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# Sample dataset for testing
@pytest.fixture
def sample_data():
    data = {
        "text": [
            "I love this product!",             # Positive
            "Worst experience ever.",           # Negative
            "It was okay, nothing special.",    # Neutral
            "Not bad at all.",                  # Positive
            "Terrible, I hate it."              # Negative
        ],
        "category": [1, -1, 0, 1, -1],  # Sentiment labels
    }
    return pd.DataFrame(data)


# Preprocessing and vectorization
@pytest.fixture
def preprocessed_data(sample_data):
    vectorizer = TfidfVectorizer(max_features=11000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(sample_data["text"])
    y = sample_data["category"]
    return X, y


def test_model_training(preprocessed_data):
    X, y = preprocessed_data

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Assert that the model was trained successfully
    assert hasattr(model, "classes_"), "The model was not trained properly. Missing 'classes_' attribute."
    assert len(model.classes_) == 3, f"The model does not recognize all three sentiment classes. Found: {model.classes_}"
