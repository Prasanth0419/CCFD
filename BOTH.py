import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv('creditcard.csv')

# Data preprocessing
data = data.sample(frac=0.1, random_state=42)  # Reduce dataset size for faster execution
labels = data['Class']
data = data.drop(columns=['Class'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Isolation Forest Model
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(X_train)
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]  # Convert predictions to 0 (normal) and 1 (fraud)

# Local Outlier Factor Model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
y_pred_lof = lof.fit_predict(X_test)
y_pred_lof = [1 if x == -1 else 0 for x in y_pred_lof]  # Convert predictions to 0 (normal) and 1 (fraud)

# Evaluate models
print("Isolation Forest:")
print(classification_report(y_test, y_pred_iso))
print("Accuracy:", accuracy_score(y_test, y_pred_iso))

print("\nLocal Outlier Factor:")
print(classification_report(y_test, y_pred_lof))
print("Accuracy:", accuracy_score(y_test, y_pred_lof))

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(y_pred_iso, bins=2, kde=False)
plt.title("Isolation Forest Predictions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.histplot(y_pred_lof, bins=2, kde=False)
plt.title("Local Outlier Factor Predictions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()
plt.show()