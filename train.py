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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

load_dotenv() 

os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/model.joblib"
MLFLOW_EXPERIMENT_NAME = "Iris_Hyper_Tuning"

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

data = pd.read_csv('./data/data.csv')
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)

X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

labels = data['species'].unique()
input_example = X_train.iloc[:5]

search_space = {
    'max_depth': hp.choice('max_depth', [2, 5, 10]),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.5)
}

def objective(params):
    
    with mlflow.start_run(run_name="Hyperopt_Run", nested=True):
        
        mlflow.log_params(params)

        mod_dt = DecisionTreeClassifier(
            max_depth=int(params['max_depth']),
            min_samples_leaf=params['min_samples_leaf'],
            random_state=1
        )
        mod_dt.fit(X_train, y_train)

        prediction = mod_dt.predict(X_test)
        accuracy = metrics.accuracy_score(prediction, y_test)
        mlflow.log_metric("accuracy", accuracy)

        cm = metrics.confusion_matrix(y_test, prediction, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"CM (depth={params['max_depth']}, leaf={params['min_samples_leaf']:.2f})")
        
        cm_plot_path = "confusion_matrix.png"
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

        loss = 1 - accuracy
        
        return {'loss': loss, 'status': STATUS_OK, 'model': mod_dt, 'accuracy': accuracy}

with mlflow.start_run(run_name="Hyperopt_Parent_Run") as parent_run:
    mlflow.set_tag("author", "Piyush")
    
    trials = Trials()
    
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )
    
    best_trial = max(trials.results, key=lambda x: x['accuracy'])
    
    mlflow.log_metric("best_accuracy", best_trial['accuracy'])
    mlflow.log_params(best_params)
    
    best_model = best_trial['model']
    
    if best_model is not None:
        joblib.dump(best_model, model_path)