import joblib


def load_model(model_path, vectorizer_path):
    """Loads the trained model and vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def make_prediction(text, model, vectorizer):
    """Makes predictions on input text using the trained model."""
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]
