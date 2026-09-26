"""
End-to-End ML Pipeline: Titanic Survival Prediction
=====================================================
Problem type: Binary Classification
Goal: Predict whether a passenger survived the Titanic disaster based on
their ticket class, sex, age, family size, fare, and port of embarkation.

This script covers:
  1. Data loading
  2. Data preprocessing (missing values, encoding, scaling, feature engineering)
  3. Exploratory Data Analysis (EDA) with saved plots
  4. Model development (Logistic Regression vs Random Forest, compared)
  5. Evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
  6. Saving the final trained pipeline (preprocessing + model together) with joblib
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

os.makedirs("plots", exist_ok=True)
os.makedirs("model", exist_ok=True)

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
df = pd.read_csv("titanic.csv")
print("Raw shape:", df.shape)
print(df.isna().sum())

# --------------------------------------------------------------------------
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# --------------------------------------------------------------------------
# Drop duplicate rows if any
df = df.drop_duplicates()

# Drop columns that are redundant or too sparse to be useful:
#  - 'deck' has too many missing values (>75%)
#  - 'class', 'who', 'adult_male', 'alive', 'embark_town', 'alone' duplicate
#    information already captured by other columns we keep/engineer
df = df.drop(columns=["deck", "class", "who", "adult_male", "alive",
                       "embark_town", "alone"])

# Feature engineering: family size = siblings/spouses + parents/children + self
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Target and features
target = "survived"
y = df[target]
X = df.drop(columns=[target])

numeric_features = ["age", "fare", "family_size", "sibsp", "parch"]
categorical_features = ["pclass", "sex", "embarked"]

print("\nFinal feature set:", numeric_features + categorical_features)

# --------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------------------
sns.set_style("whitegrid")

# Survival rate by sex
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("plots/survival_by_sex.png", dpi=120)
plt.close()

# Survival rate by passenger class
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("plots/survival_by_class.png", dpi=120)
plt.close()

# Age distribution split by survival
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="age", hue="survived", kde=True, bins=30, multiple="stack")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("plots/age_distribution.png", dpi=120)
plt.close()

# Correlation heatmap of numeric features
plt.figure(figsize=(6, 5))
num_df = df[numeric_features + [target]].copy()
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=120)
plt.close()

print("\nEDA plots saved to plots/")

# --------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------------------------------
# 5. PREPROCESSING PIPELINE (bundled with the model so it's saved together)
# --------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# --------------------------------------------------------------------------
# 6. MODEL DEVELOPMENT — compare Logistic Regression vs Random Forest
# --------------------------------------------------------------------------
results = {}

# --- Logistic Regression ---
logreg_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
logreg_pipe.fit(X_train, y_train)
pred = logreg_pipe.predict(X_test)
proba = logreg_pipe.predict_proba(X_test)[:, 1]
results["LogisticRegression"] = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
}

# --- Random Forest (with a small hyperparameter search) ---
rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [4, 6, 8, None],
    "classifier__min_samples_leaf": [1, 2, 4],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
pred = best_rf.predict(X_test)
proba = best_rf.predict_proba(X_test)[:, 1]
results["RandomForest"] = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
}
print("\nBest RF params:", grid.best_params_)

# --------------------------------------------------------------------------
# 7. EVALUATION — pick the best model by ROC-AUC
# --------------------------------------------------------------------------
print("\n=== Model comparison ===")
for name, m in results.items():
    print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))

best_name = max(results, key=lambda n: results[n]["roc_auc"])
best_model = logreg_pipe if best_name == "LogisticRegression" else best_rf
print(f"\nBest model: {best_name}")

final_pred = best_model.predict(X_test)
final_proba = best_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=120)
plt.close()

plt.figure(figsize=(5, 5))
RocCurveDisplay.from_predictions(y_test, final_proba)
plt.title(f"ROC Curve — {best_name}")
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=120)
plt.close()

print("\nClassification report:\n", classification_report(y_test, final_pred))

# --------------------------------------------------------------------------
# 8. SAVE THE FINAL TRAINED PIPELINE (preprocessing + model bundled together)
# --------------------------------------------------------------------------
joblib.dump(best_model, "model/titanic_model.joblib")

metadata = {
    "best_model": best_name,
    "metrics": results[best_name],
    "all_results": results,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved trained pipeline to model/titanic_model.joblib")
print("Saved metadata to model/metadata.json")
