import pandas as pd
import pytest


@pytest.fixture(scope='module')
def load_data():
    try:
        df = pd.read_csv('artifacts/data.csv')
    except FileNotFoundError:
        pytest.fail("The 'artifacts/data.csv' file was not found. Ensure the file path is correct.")
    return df


def test_is_not_empty(load_data):
    assert not load_data.empty, "The DataFrame is empty!"


def test_column_existence(load_data):
    required_columns = ['content', 'url']
    missing_columns = [col for col in required_columns if col not in load_data.columns]
    assert not missing_columns, f"Missing columns in the dataset: {', '.join(missing_columns)}"


def test_no_missing_values(load_data):
    missing_values = load_data.isnull().sum()
    assert missing_values.sum() == 0, f"Missing values found:\n{missing_values[missing_values > 0]}"


def test_number_of_rows(load_data):
    min_rows = 1000
    assert load_data.shape[0] > min_rows, f"Not enough rows in the dataset: {load_data.shape[0]} (minimum required: {min_rows})"
