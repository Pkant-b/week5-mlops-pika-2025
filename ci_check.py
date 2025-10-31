import os
import sys
import mlflow
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import joblib  # <--- Make sure this is in requirements.txt

# Load .env file (for local testing)
load_dotenv()

# --- Configuration ---
MODEL_NAME = "iris_decision_tree"
ARTIFACT_DIR = "artifacts"
DATA_PATH = "data/data.csv"
MIN_ACCURACY = 0.90

# Create artifact directory if it doesn't exist
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- 1. Connect to MLflow ---
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    print("MLFLOW_TRACKING_URI environment variable not set.")
    sys.exit(1)

mlflow.set_tracking_uri(tracking_uri)
client = mlflow.tracking.MlflowClient()

try:
    # --- 2. Get Latest Model Version (Logic from your reference) ---
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

    # --- 3. Run Sanity Check (The Core Task) ---
    
    # --- Download the model artifacts (Logic from your reference) ---
    print(f"Downloading model artifacts from run: {latest_version.run_id}")
    
    # This downloads the 'model' folder (logged in train.py) into the 'ARTIFACT_DIR'
    mlflow.artifacts.download_artifacts(
        run_id=latest_version.run_id,
        artifact_path="model", # The artifact_path from train.py
        dst_path=ARTIFACT_DIR
    )
    
    # --- Load the model from the downloaded file ---
    # The default path MLflow saves to is 'artifacts/model/model.pkl'
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
    
    # The Sanity Check Assertion
    assert accuracy > MIN_ACCURACY, f"Model accuracy {accuracy:.4f} is below threshold {MIN_ACCURACY}"
    print("Sanity check passed!")

    # --- 4. Get Data for CML Report ---
    print("Fetching run data for CML report...")
    run = client.get_run(latest_version.run_id)
    metrics = run.data.metrics
    
    # Download the confusion matrix plot
    cm_artifact_path = "confusion_matrix.png" # Must match name in train.py
    local_cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    
    client.download_artifacts(
        run_id=latest_version.run_id,
        path=cm_artifact_path,
        dst_path=ARTIFACT_DIR
    )
    print(f"Downloaded confusion matrix to {local_cm_path}")

    # --- 5. Generate CML Report (to stdout) ---
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

    # Print the full report as Markdown
    print("\n## Model Sanity Check Report")
    print(f"**Model:** `{MODEL_NAME}` (Version {latest_version.version})")
    print(f"**Run ID:** `{latest_version.run_id}`")
    print("\n### Pytest Sanity Check")
    print(f"✅ **PASSED**: Accuracy ({accuracy:.4f}) is above threshold ({MIN_ACCURACY})")
    print("\n### Model Metrics")
    print(metrics_table_string)
    print("\n### Confusion Matrix")
    # We must use the local path for the CML report
    print(f"![Confusion Matrix]({local_cm_path})")

except Exception as e:
    print(f"CI Check Failed: {e}")
    sys.exit(1)