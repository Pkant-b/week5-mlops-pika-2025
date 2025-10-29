import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

print("=== IRIS Classification Training ===")

os.makedirs("artifacts", exist_ok=True)

print("\nLoading dataset...")
data = pd.read_csv('./data/data.csv')
print(f"Dataset loaded: {len(data)} samples")
print(f"Features: {list(data.columns[:-1])}")
print(f"Classes: {list(data['species'].unique())}")
print(f"Class distribution:\n{data['species'].value_counts()}")

print("\nSplitting data...")
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

print(f"Training samples: {len(train)}")
print(f"Test samples: {len(test)}")

print("\nTraining Decision Tree...")
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)
print("Model trained successfully")

prediction = mod_dt.predict(X_test)
accuracy = metrics.accuracy_score(prediction, y_test)
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.3f}")

model_path = "artifacts/model.joblib"
joblib.dump(mod_dt, model_path)
print(f"\nModel saved to: {model_path}")
