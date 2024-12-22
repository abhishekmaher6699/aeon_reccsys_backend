import pytest
import pickle
import os
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def load_pickle():
    pickle_file = './artifacts/objects.pkl'
    
    if not os.path.exists(pickle_file):
        pytest.fail(f"The pickle file {pickle_file} does not exist.")
    
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    return data

def test_pickle_integrity(load_pickle):
    required_keys = ["df", "url_to_id", "vectorizer", "tfidf_matrix", "pca", "pca_sim_scores"]
    missing_keys = [key for key in required_keys if key not in load_pickle]
    
    assert not missing_keys, f"Missing keys in the pickle data: {', '.join(missing_keys)}"


def test_dataframe_content(load_pickle):
    df = load_pickle.get("df", None)
    assert df is not None, "DataFrame 'df' is missing in the pickle file."
    
    assert not df.empty, "DataFrame 'df' is empty."
    
    required_columns = ["content", "url"]
    for column in required_columns:
        assert column in df.columns, f"Column '{column}' is missing in the DataFrame."


def test_url_to_id_mapping(load_pickle):
    url_to_id = load_pickle.get("url_to_id", None)
    assert len(url_to_id) > 0, "'url_to_id' mapping is empty."


def test_vectorizer_type(load_pickle):
    vectorizer = load_pickle.get("vectorizer", None)
    assert vectorizer is not None, "'vectorizer' is missing in the pickle file."
    assert hasattr(vectorizer, "transform"), "'vectorizer' does not have a 'transform' method."


def test_tfidf_matrix_shape(load_pickle):
    tfidf_matrix = load_pickle.get("tfidf_matrix", None)
    assert tfidf_matrix is not None, "'tfidf_matrix' is missing in the pickle file."
    assert tfidf_matrix.shape[0] > 0 and tfidf_matrix.shape[1] > 0, "TF-IDF matrix has invalid dimensions."


def test_pca_object(load_pickle):
    pca = load_pickle.get("pca", None)
    assert pca is not None, "'pca' is missing in the pickle file."
    assert hasattr(pca, "transform"), "'pca' does not have a 'transform' method."


def test_pca_sim_scores(load_pickle):
    pca_sim_scores = load_pickle.get("pca_sim_scores", None)
    assert pca_sim_scores is not None, "'pca_sim_scores' is missing in the pickle file."
    assert isinstance(pca_sim_scores, np.ndarray), "'pca_sim_scores' is not a list."
    assert len(pca_sim_scores) > 0, "'pca_sim_scores' is empty."
