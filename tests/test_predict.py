import pytest
from src.predict import load_model, make_prediction

import json
from app import app


def test_positive_sentiment():
    with app.test_client() as client:
        response = client.post(
            "/predict",
            data=json.dumps({"text": "I love a red flower!"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert response.status_code == 200
        assert data["prediction"] == "Positive"


def test_negative_sentiment():
    with app.test_client() as client:
        response = client.post(
            "/predict",
            data=json.dumps({"text": "I hate a red flower!"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert response.status_code == 200
        assert data["prediction"] == "Negative"


def test_neutral_sentiment():
    with app.test_client() as client:
        response = client.post(
            "/predict",
            data=json.dumps({"text": "This is a red flower!"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert response.status_code == 200
        assert data["prediction"] == "Neutral"


@pytest.fixture
def model_and_vectorizer():
    model, vectorizer = load_model('models/best_naive_bayes_model.pkl', 'models/vectorizer.pkl')
    return model, vectorizer


def test_make_prediction(model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    prediction = make_prediction("This is a great product!", model, vectorizer)
    assert prediction in [-1, 0, 1]
