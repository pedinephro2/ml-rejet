"""
Generate synthetic pediatric kidney graft rejection dataset.
500 patients with realistic clinical distributions.
Target: rejet (1) / pas de rejet (0) — ~25% rejection rate.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 500

# --- Generate base features ---
age_mois = np.random.randint(12, 216, N)  # 1-18 years in months
poids_kg = np.clip(age_mois * 0.15 + np.random.normal(0, 3, N), 8, 80)
jours_post_greffe = np.random.randint(7, 3650, N)  # 1 week to 10 years

# Normal kidney function values (pediatric ranges)
creatinine_umoll = np.random.normal(80, 30, N)
uree_mmoll = np.random.normal(6, 2, N)
proteinurie_24h = np.abs(np.random.exponential(100, N))
dfg_ml_min = np.random.normal(90, 20, N)
hemoglobine_gdl = np.random.normal(12, 1.5, N)
crp_mgl = np.abs(np.random.exponential(3, N))

# Tacrolimus trough level (therapeutic range 5-15 ng/mL)
tacrolimus_taux_ngml = np.random.normal(8, 3, N)

# Blood pressure (pediatric)
pression_arterielle_sys = np.random.normal(110, 12, N)
pression_arterielle_dia = np.random.normal(70, 8, N)

# Banff score: 0=normal, 1=borderline, 2=acute rejection, 3=chronic rejection
score_banff = np.random.choice([0, 1, 2, 3], N, p=[0.55, 0.20, 0.15, 0.10])

# Creatinine variation vs previous (%)
variation_creatinine = np.random.normal(5, 15, N)

# --- Generate rejection label with realistic correlations ---
rejection_score = np.zeros(N)

# High Banff score strongly increases rejection risk
rejection_score += (score_banff == 2) * 3.0
rejection_score += (score_banff == 3) * 4.0
rejection_score += (score_banff == 1) * 1.0

# Low tacrolimus → more rejection (under-immunosuppression)
rejection_score += (tacrolimus_taux_ngml < 5) * 2.0
rejection_score += (tacrolimus_taux_ngml < 3) * 1.5

# High creatinine → kidney dysfunction
rejection_score += (creatinine_umoll > 120) * 1.5
rejection_score += (creatinine_umoll > 150) * 1.0

# High creatinine variation → acute change
rejection_score += (variation_creatinine > 20) * 1.5
rejection_score += (variation_creatinine > 40) * 1.0

# Low GFR
rejection_score += (dfg_ml_min < 60) * 1.5
rejection_score += (dfg_ml_min < 40) * 1.0

# High proteinuria
rejection_score += (proteinurie_24h > 300) * 1.0
rejection_score += (proteinurie_24h > 500) * 1.0

# High CRP (inflammation)
rejection_score += (crp_mgl > 10) * 1.0

# Hypertension
rejection_score += (pression_arterielle_sys > 130) * 0.5

# Early post-transplant period is higher risk
rejection_score += (jours_post_greffe < 90) * 1.0

# Convert to probability and generate binary label
noise = np.random.normal(0, 1.5, N)
rejection_prob = 1 / (1 + np.exp(-(rejection_score + noise - 4)))
rejet = (rejection_prob > 0.5).astype(int)

# Adjust values for rejection patients to make correlations stronger
for i in range(N):
    if rejet[i] == 1:
        creatinine_umoll[i] += np.random.uniform(20, 60)
        uree_mmoll[i] += np.random.uniform(2, 5)
        proteinurie_24h[i] += np.random.uniform(100, 400)
        dfg_ml_min[i] -= np.random.uniform(10, 30)
        crp_mgl[i] += np.random.uniform(5, 15)
        variation_creatinine[i] += np.random.uniform(10, 30)
        hemoglobine_gdl[i] -= np.random.uniform(0.5, 2)

# Clip to realistic ranges
creatinine_umoll = np.clip(creatinine_umoll, 30, 500).round(1)
uree_mmoll = np.clip(uree_mmoll, 1.5, 30).round(1)
proteinurie_24h = np.clip(proteinurie_24h, 0, 3000).round(0)
tacrolimus_taux_ngml = np.clip(tacrolimus_taux_ngml, 0.5, 25).round(1)
pression_arterielle_sys = np.clip(pression_arterielle_sys, 70, 180).round(0).astype(int)
pression_arterielle_dia = np.clip(pression_arterielle_dia, 40, 120).round(0).astype(int)
dfg_ml_min = np.clip(dfg_ml_min, 5, 150).round(1)
hemoglobine_gdl = np.clip(hemoglobine_gdl, 5, 17).round(1)
crp_mgl = np.clip(crp_mgl, 0.1, 100).round(1)
variation_creatinine = np.clip(variation_creatinine, -30, 100).round(1)
poids_kg = poids_kg.round(1)

# --- Build DataFrame ---
df = pd.DataFrame({
    "creatinine_umoll": creatinine_umoll,
    "uree_mmoll": uree_mmoll,
    "proteinurie_24h": proteinurie_24h,
    "tacrolimus_taux_ngml": tacrolimus_taux_ngml,
    "pression_arterielle_sys": pression_arterielle_sys,
    "pression_arterielle_dia": pression_arterielle_dia,
    "score_banff": score_banff,
    "age_mois": age_mois,
    "poids_kg": poids_kg,
    "jours_post_greffe": jours_post_greffe,
    "dfg_ml_min": dfg_ml_min,
    "hemoglobine_gdl": hemoglobine_gdl,
    "crp_mgl": crp_mgl,
    "variation_creatinine": variation_creatinine,
    "rejet": rejet,
})

output_path = os.path.join(os.path.dirname(__file__), "dataset_rejet_greffe.csv")
df.to_csv(output_path, index=False)

print(f"Dataset generated: {output_path}")
print(f"Total patients: {len(df)}")
print(f"Rejection rate: {rejet.mean()*100:.1f}%")
print(f"\nClass distribution:")
print(df["rejet"].value_counts())
print(f"\nFeature summary:")
print(df.describe().round(2))
