from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def test_model_training():
    # Mock data
    X = [[0, 1, 2], [1, 0, 3]]
    y = [-1, 0, 1]
    model = MultinomialNB()
    model.fit(X, y)
    assert model.predict([[0, 1, 2]]) == [0]
