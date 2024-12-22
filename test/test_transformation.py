import pytest
import pandas as pd
from io import StringIO
from src.data_transformation import expand_words, remove_punctuations, remove_stopwords, clean_data

@pytest.mark.parametrize(
    "text, expected_output",
    [
        ("I can't do this", "I cannot do this"),
        ("It's a beautiful day.", "It is a beautiful day."),
        ("You'll be fine.", "You will be fine."),
    ],
)
def test_expand_words(text, expected_output):
    assert expand_words(text) == expected_output


@pytest.mark.parametrize(
    "text, expected_output",
    [
        ("Hello, world! How's it going?", "Hello world Hows it going"),
        ("No punctuations here", "No punctuations here"),
        ("Special @#characters!!!", "Special characters"),
    ],
)
def test_remove_punctuations(text, expected_output):
    assert remove_punctuations(text) == expected_output


@pytest.mark.parametrize(
    "text, expected_output",
    [
        ("This is a simple test", "simple test"),
        ("Stopwords should be removed", "Stopwords removed"),
        ("Keep the important words only", "Keep important words"),
    ],
)
def test_remove_stopwords(text, expected_output):
    assert remove_stopwords(text) == expected_output