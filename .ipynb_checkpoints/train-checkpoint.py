import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv() 

os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/model.joblib"

mlflow.set_experiment("Iris_Decision_Tree_Tuning")

data = pd.read_csv('./data/data.csv')
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)

X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

labels = data['species'].unique()
input_example = X_train.iloc[:5]
max_depth_params = [2, 5, 10]

best_accuracy = 0.0
best_model = None

with mlflow.start_run(run_name="Decision_Tree_HPT") as parent_run:
    mlflow.set_tag("author", "Piyush")
    mlflow.log_param("hyperparameter_list", max_depth_params)

    for depth in max_depth_params:
        with mlflow.start_run(run_name=f"run_depth_{depth}", nested=True) as child_run:
            
            mlflow.log_param("max_depth", depth)

            mod_dt = DecisionTreeClassifier(max_depth=depth, random_state=1)
            mod_dt.fit(X_train, y_train)

            prediction = mod_dt.predict(X_test)
            accuracy = metrics.accuracy_score(prediction, y_test)
            
            mlflow.log_metric("accuracy", accuracy)

            cm = metrics.confusion_matrix(y_test, prediction, labels=labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix (max_depth={depth})')
            
            cm_plot_path = f"confusion_matrix_depth_{depth}.png"
            plt.savefig(cm_plot_path)
            plt.close()
            
            mlflow.log_artifact(cm_plot_path)
            os.remove(cm_plot_path)

            mlflow.sklearn.log_model(
                sk_model=mod_dt,
                artifact_path="model",
                registered_model_name="iris_decision_tree",
                input_example=input_example
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = mod_dt

    mlflow.log_metric("best_accuracy", best_accuracy)

if best_model is not None:
    joblib.dump(best_model, model_path)