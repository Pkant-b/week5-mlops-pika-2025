import pandas as pd
import pytest

def test_data_columns():
    data_path = 'data/data.csv'
    df = pd.read_csv(data_path)

    expected_columns = [
        "sepal_length", 
        "sepal_width", 
        "petal_length", 
        "petal_width",
        "species" 
    ]
    assert all([col in df.columns for col in expected_columns])


def test_data_types():
    data_path = 'data/data.csv'
    df = pd.read_csv(data_path)

    feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in feature_columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' doesn't contain numbers."

        
