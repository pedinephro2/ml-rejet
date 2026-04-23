"""
Train kidney graft rejection prediction model.
Tests: RandomForest, GradientBoosting, XGBoost, LogisticRegression.
Handles class imbalance with SMOTE + class_weight.
5-fold cross-validation. SHAP explainability.
"""
import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
    recall_score, precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "dataset_rejet_greffe.csv")
MODEL_DIR = BASE_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
print("=" * 60)
print("KIDNEY GRAFT REJECTION MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset: {len(df)} patients")
print(f"Rejection rate: {df['rejet'].mean()*100:.1f}%")

X = df.drop(columns=["rejet"])
y = df["rejet"]
feature_names = list(X.columns)

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print(f"After SMOTE: {len(X_resampled)} samples")
print(f"  - No rejection: {(y_resampled == 0).sum()}")
print(f"  - Rejection: {(y_resampled == 1).sum()}")

# Define models
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=3, eval_metric="logloss", random_state=42
    ),
}

# 5-Fold Cross-Validation
print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

scoring = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
}

for name, model in models.items():
    cv_results = cross_validate(
        model, X_resampled, y_resampled,
        cv=cv, scoring=scoring, return_train_score=False
    )

    metrics = {
        "accuracy": float(np.mean(cv_results["test_accuracy"])),
        "roc_auc": float(np.mean(cv_results["test_roc_auc"])),
        "f1": float(np.mean(cv_results["test_f1"])),
        "sensitivity": float(np.mean(cv_results["test_recall"])),
        "precision": float(np.mean(cv_results["test_precision"])),
    }
    # Specificity = TN / (TN + FP), approximate from accuracy and sensitivity
    metrics["specificity"] = float(
        2 * metrics["accuracy"] - metrics["sensitivity"]
        if metrics["sensitivity"] <= metrics["accuracy"] * 2
        else metrics["accuracy"]
    )
    results[name] = metrics

    print(f"\n{name}:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:     {metrics['roc_auc']:.4f}")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")

# Select best model by AUC-ROC
best_name = max(results, key=lambda k: results[k]["roc_auc"])
print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best_name} (AUC-ROC: {results[best_name]['roc_auc']:.4f})")
print(f"{'=' * 60}")

# Train best model on full resampled data
best_model = models[best_name]
best_model.fit(X_resampled, y_resampled)

# Evaluate on original data (no SMOTE)
y_pred = best_model.predict(X_scaled)
y_proba = best_model.predict_proba(X_scaled)[:, 1]

print(f"\nPerformance on original data:")
print(f"  Accuracy:    {accuracy_score(y, y_pred):.4f}")
print(f"  AUC-ROC:     {roc_auc_score(y, y_proba):.4f}")
print(f"  F1-Score:    {f1_score(y, y_pred):.4f}")
print(f"  Sensitivity: {recall_score(y, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y, y_pred, target_names=["No Rejection", "Rejection"]))
print(f"Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# SHAP explainability
print(f"\nComputing SHAP values...")
if best_name in ["RandomForest", "GradientBoosting", "XGBoost"]:
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.LinearExplainer(best_model, X_scaled)

shap_values = explainer.shap_values(X_scaled)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # class 1 (rejection)

mean_shap = np.abs(shap_values).mean(axis=0)
# Ensure mean_shap values are scalars
mean_shap_list = [float(v) if np.ndim(v) == 0 else float(np.mean(v)) for v in mean_shap]
feature_importance = sorted(
    zip(feature_names, mean_shap_list), key=lambda x: x[1], reverse=True
)

print("\nTop feature importance (SHAP):")
for fname, imp in feature_importance[:5]:
    print(f"  {fname}: {imp:.4f}")

# Save everything
model_path = os.path.join(MODEL_DIR, "model_rejet.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler_rejet.pkl")
features_path = os.path.join(MODEL_DIR, "features_rejet.json")
metrics_path = os.path.join(MODEL_DIR, "metrics_rejet.json")

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

with open(features_path, "w") as f:
    json.dump(feature_names, f, indent=2)

metrics_output = {
    "best_model": best_name,
    "cv_results": results,
    "feature_importance": [
        {"feature": fname, "shap_importance": round(float(imp), 4)}
        for fname, imp in feature_importance
    ],
    "original_data_metrics": {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_proba)), 4),
        "f1": round(float(f1_score(y, y_pred)), 4),
        "sensitivity": round(float(recall_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred)), 4),
    },
}

with open(metrics_path, "w") as f:
    json.dump(metrics_output, f, indent=2)

print(f"\nFiles saved:")
print(f"  Model:    {model_path}")
print(f"  Scaler:   {scaler_path}")
print(f"  Features: {features_path}")
print(f"  Metrics:  {metrics_path}")
print(f"\nDone!")
