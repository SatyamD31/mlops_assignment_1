import pytest
from src.data_preprocessing import preprocess_text


def test_preprocess_text():
    text = "Check out https://example.com @user #hashtag!"
    expected_output = "check"
    assert preprocess_text(text) == expected_output
