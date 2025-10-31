import os
import sys
import mlflow
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import joblib 

load_dotenv()

MODEL_NAME = "iris_decision_tree"
ARTIFACT_DIR = "artifacts"
DATA_PATH = "data/data.csv"
MIN_ACCURACY = 0.90

os.makedirs(ARTIFACT_DIR, exist_ok=True)

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    print("MLFLOW_TRACKING_URI environment variable not set.")
    sys.exit(1)

mlflow.set_tracking_uri(tracking_uri)
client = mlflow.tracking.MlflowClient()

try:
    print(f"Fetching latest version of model: {MODEL_NAME}")
    versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'",
        order_by=["version_number DESC"],
        max_results=1
    )
    if not versions:
        print(f"No model versions found for model: {MODEL_NAME}")
        sys.exit(1)
    
    latest_version = versions[0]
    print(f"Found version: {latest_version.version}, Run ID: {latest_version.run_id}")

    
    print(f"Downloading model artifacts from run: {latest_version.run_id}")
    
    mlflow.artifacts.download_artifacts(
        run_id=latest_version.run_id,
        artifact_path="model", # The artifact_path from train.py
        dst_path=ARTIFACT_DIR
    )
    
  
    model_path = os.path.join(ARTIFACT_DIR, "model", "model.pkl")
    if not os.path.exists(model_path):
        print(f"Failed to download model. Expected file not found at: {model_path}")
        sys.exit(1)
        
    print("Loading model for sanity check...")
    model = joblib.load(model_path)
    print("Model loaded successfully from downloaded artifacts.")

    # --- Run the sanity check ---
    print("Loading test data...")
    df = pd.read_csv(DATA_PATH)
    X_test = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_test = df["species"]

    print("Running predictions...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")
    
    assert accuracy > MIN_ACCURACY, f"Model accuracy {accuracy:.4f} is below threshold {MIN_ACCURACY}"
    print("Sanity check passed!")

    print("Fetching run data for CML report...")
    run = client.get_run(latest_version.run_id)
    metrics = run.data.metrics
    
    cm_artifact_path = "confusion_matrix.png" # Must match name in train.py
    local_cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    
    client.download_artifacts(
        run_id=latest_version.run_id,
        path=cm_artifact_path,
        dst_path=ARTIFACT_DIR
    )
    print(f"Downloaded confusion matrix to {local_cm_path}")

    header = ["Metric", "Value"]
    table_data = []
    for key, val in metrics.items():
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        table_data.append([key, val_str])

    metrics_table_string = tabulate(
        table_data, 
        headers=header, 
        tablefmt="github"
    )

except Exception as e:
    print(f"CI Check Failed: {e}")
    sys.exit(1)