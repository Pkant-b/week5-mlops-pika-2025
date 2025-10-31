from dotenv import load_dotenv
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os 


load_dotenv()

def test_model_performance():
    df = pd.read_csv('data/data.csv')


    model_uri = "models:/iris_decision_tree/latest"
    print(f"Loading model from: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        pytest.fail(f"Failed to load model from MLflow Registry: {e}")
    
    print("Model loaded successfully from MLflow.")
    
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)
    print(f"Model Accuracy: {accuracy:.3f}")
    assert accuracy > 0.90, f"Model accuracy {accuracy:.2f} is below the threshold of 0.90."
    
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Loaded from MLflow)')

    plt.savefig('matrix/confusion_matrix_mlflow.png')
    print("Confusion matrix saved to 'confusion_matrix_mlflow.png'")