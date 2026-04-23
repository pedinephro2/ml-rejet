"""
FastAPI microservice for kidney graft rejection prediction.
Port: 8001
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import json
import os
import shap

app = FastAPI(
    title="Kidney Graft Rejection Prediction API",
    description="Pediatric nephrology - graft rejection risk prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

model = joblib.load(os.path.join(MODEL_DIR, "model_rejet.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_rejet.pkl"))

with open(os.path.join(MODEL_DIR, "features_rejet.json")) as f:
    feature_names = json.load(f)

with open(os.path.join(MODEL_DIR, "metrics_rejet.json")) as f:
    metrics_data = json.load(f)

# SHAP explainer
try:
    explainer = shap.TreeExplainer(model)
except Exception:
    explainer = None


# --- Pydantic Models ---

class PatientInput(BaseModel):
    creatinine_umoll: float = Field(..., ge=10, le=1000, description="Serum creatinine (µmol/L)")
    uree_mmoll: float = Field(..., ge=0.5, le=50, description="Blood urea (mmol/L)")
    proteinurie_24h: float = Field(..., ge=0, le=10000, description="24h proteinuria (mg/24h)")
    tacrolimus_taux_ngml: float = Field(..., ge=0, le=50, description="Tacrolimus trough level (ng/mL)")
    pression_arterielle_sys: float = Field(..., ge=60, le=220, description="Systolic BP (mmHg)")
    pression_arterielle_dia: float = Field(..., ge=30, le=140, description="Diastolic BP (mmHg)")
    score_banff: int = Field(..., ge=0, le=3, description="Banff biopsy score (0-3)")
    age_mois: int = Field(..., ge=1, le=300, description="Age in months")
    poids_kg: float = Field(..., ge=2, le=150, description="Weight (kg)")
    jours_post_greffe: int = Field(..., ge=1, le=10000, description="Days since transplant")
    dfg_ml_min: float = Field(..., ge=1, le=200, description="GFR (mL/min/1.73m²)")
    hemoglobine_gdl: float = Field(..., ge=3, le=20, description="Hemoglobin (g/dL)")
    crp_mgl: float = Field(..., ge=0, le=300, description="CRP (mg/L)")
    variation_creatinine: float = Field(..., ge=-50, le=200, description="Creatinine variation vs previous (%)")


class PredictionResponse(BaseModel):
    risque_rejet_pct: float
    classe_risque: str
    facteurs_cles: List[dict]
    alerte: Optional[str]
    modele_utilise: str


class BatchInput(BaseModel):
    patients: List[PatientInput]


# --- Helper functions ---

def classify_risk(risk_pct: float) -> str:
    if risk_pct < 25:
        return "FAIBLE"
    elif risk_pct < 50:
        return "MODERE"
    elif risk_pct < 75:
        return "ELEVE"
    else:
        return "CRITIQUE"


def generate_alert(risk_pct: float, classe: str) -> Optional[str]:
    if risk_pct >= 75:
        return f"ALERTE CRITIQUE: Risque de rejet très élevé ({risk_pct}%). Biopsie urgente et ajustement immunosuppresseur recommandés."
    elif risk_pct >= 50:
        return f"ALERTE: Risque de rejet élevé ({risk_pct}%). Surveillance renforcée et contrôle biologique recommandés."
    return None


def predict_single(patient: PatientInput) -> PredictionResponse:
    # Build feature array
    features = np.array([[
        patient.creatinine_umoll,
        patient.uree_mmoll,
        patient.proteinurie_24h,
        patient.tacrolimus_taux_ngml,
        patient.pression_arterielle_sys,
        patient.pression_arterielle_dia,
        patient.score_banff,
        patient.age_mois,
        patient.poids_kg,
        patient.jours_post_greffe,
        patient.dfg_ml_min,
        patient.hemoglobine_gdl,
        patient.crp_mgl,
        patient.variation_creatinine,
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    risk_pct = round(float(model.predict_proba(features_scaled)[0][1]) * 100, 1)
    classe = classify_risk(risk_pct)
    alerte = generate_alert(risk_pct, classe)

    # SHAP explanation (top 5 factors)
    facteurs_cles = []
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(features_scaled)
            if isinstance(shap_vals, list):
                sv = np.array(shap_vals[1]).flatten()
            elif isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 3:
                    sv = shap_vals[1, 0, :]
                elif shap_vals.ndim == 2:
                    sv = shap_vals[0, :]
                else:
                    sv = shap_vals.flatten()
            else:
                sv = np.array(shap_vals).flatten()
            shap_dict = list(zip(feature_names, sv.tolist()))
            shap_sorted = sorted(shap_dict, key=lambda x: abs(x[1]), reverse=True)[:5]
            facteurs_cles = [
                {
                    "feature": name,
                    "impact": round(float(val), 4),
                    "direction": "augmente le risque" if val > 0 else "diminue le risque",
                }
                for name, val in shap_sorted
            ]
        except Exception:
            pass
    if not facteurs_cles:
        # Fallback: use feature importances if available
        if hasattr(model, "feature_importances_"):
            imp = sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:5]
            facteurs_cles = [
                {"feature": name, "impact": round(float(val), 4), "direction": "contribue au risque"}
                for name, val in imp
            ]

    return PredictionResponse(
        risque_rejet_pct=risk_pct,
        classe_risque=classe,
        facteurs_cles=facteurs_cles,
        alerte=alerte,
        modele_utilise=metrics_data.get("best_model", "Unknown"),
    )


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "model": metrics_data.get("best_model"), "port": 8001}


@app.get("/metrics")
def get_metrics():
    return metrics_data


@app.post("/predict/rejet", response_model=PredictionResponse)
def predict_rejet(patient: PatientInput):
    try:
        return predict_single(patient)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/rejet/batch", response_model=List[PredictionResponse])
def predict_rejet_batch(batch: BatchInput):
    try:
        return [predict_single(p) for p in batch.patients]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
