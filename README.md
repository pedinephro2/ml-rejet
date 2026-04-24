# ML Rejet — Kidney Graft Rejection Prediction

FastAPI microservice that uses an XGBoost model to predict the risk of kidney graft rejection. Provides risk stratification, SHAP-based explainability, and batch prediction support for the Nephro Platform.

## Tech Stack

- Python 3.10+
- FastAPI
- XGBoost, scikit-learn
- SHAP (model explainability)
- pandas, numpy
- Uvicorn

## Port

`8001`

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/metrics` | Model performance metrics |
| POST | `/predict/rejet` | Predict rejection risk for one patient |
| POST | `/predict/rejet/batch` | Batch predictions for multiple patients |

## Input Features

The model accepts 14 clinical parameters:

| Feature | Description |
|---|---|
| `creatinine_seriq` | Serum creatinine (µmol/L) |
| `uree_sang` | Blood urea (mmol/L) |
| `proteinurie` | Proteinuria (g/24h) |
| `tacrolimus_trough` | Tacrolimus trough level (ng/mL) |
| `tension_systolique` | Systolic blood pressure (mmHg) |
| `tension_diastolique` | Diastolic blood pressure (mmHg) |
| `score_banff` | Banff biopsy classification score |
| `age` | Patient age (years) |
| `poids` | Weight (kg) |
| `jours_post_greffe` | Days since transplant |
| `dfg` | Glomerular filtration rate (mL/min/1.73m²) |
| `hemoglobine` | Hemoglobin (g/dL) |
| `crp` | C-reactive protein (mg/L) |
| `variation_creatinine` | Creatinine variation from last measurement |

## Output

```json
{
  "risk_percentage": 72.4,
  "risk_class": "ELEVE",
  "key_risk_factors": ["creatinine_seriq", "score_banff", "variation_creatinine"],
  "alert_messages": ["High rejection risk detected — clinical review recommended"],
  "model_metadata": {
    "model_version": "1.0",
    "prediction_timestamp": "2026-04-24T10:00:00Z"
  }
}
```

### Risk Classes

| Class | Description |
|---|---|
| `FAIBLE` | Low risk (< 25%) |
| `MODERE` | Moderate risk (25–50%) |
| `ELEVE` | High risk (50–75%) |
| `CRITIQUE` | Critical risk (> 75%) |

## Features

- **XGBoost model** — trained on pediatric nephrology transplant data
- **SHAP explainability** — identifies which features contributed most to the prediction
- **Risk stratification** — four-tier classification with clinical alert messages
- **Batch processing** — predict for multiple patients in a single request
- **Feature scaling** — automatic normalization of input features

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Docker

```bash
docker build -t azizos07/ml-rejet:latest .
docker run -p 8001:8001 azizos07/ml-rejet:latest
```

## Example Request

```bash
curl -X POST http://localhost:8001/predict/rejet \
  -H "Content-Type: application/json" \
  -d '{
    "creatinine_seriq": 280,
    "uree_sang": 12.5,
    "proteinurie": 1.2,
    "tacrolimus_trough": 6.0,
    "tension_systolique": 145,
    "tension_diastolique": 92,
    "score_banff": 2,
    "age": 10,
    "poids": 35,
    "jours_post_greffe": 180,
    "dfg": 38,
    "hemoglobine": 10.2,
    "crp": 18.5,
    "variation_creatinine": 45
  }'
```
