import pandas as pd
import joblib
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def test_model_performance():
    df = pd.read_csv('data/data.csv')
    model = joblib.load('artifacts/model.joblib')

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)
    assert accuracy > 0.90, f"Model accuracy {accuracy:.2f} is below the threshold of 0.90."
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix.png')
