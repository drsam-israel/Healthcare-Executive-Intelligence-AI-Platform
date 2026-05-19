import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

N = 5000

output_path = Path("data/synthetic")
output_path.mkdir(parents=True, exist_ok=True)

admission_dates = pd.date_range(
    start="2023-01-01",
    end="2024-12-31",
    periods=N
)

departments = [
    "Emergency",
    "Internal Medicine",
    "Surgery",
    "ICU",
    "Cardiology",
    "Obstetrics",
    "Pediatrics"
]

payer_types = [
    "Private Insurance",
    "Medicare",
    "Medicaid",
    "Self-Pay",
    "Government"
]

diagnoses = [
    "Pneumonia",
    "Heart Failure",
    "Diabetes Complication",
    "Stroke",
    "Sepsis",
    "Fracture",
    "COPD",
    "Renal Failure"
]

df = pd.DataFrame({
    "patient_id": np.arange(100001, 100001 + N),
    "admission_id": np.arange(500001, 500001 + N),
    "admission_date": admission_dates,
    "department": np.random.choice(departments, N),
    "payer_type": np.random.choice(payer_types, N),
    "primary_diagnosis": np.random.choice(diagnoses, N),
    "length_of_stay": np.random.gamma(shape=2.5, scale=2.0, size=N).round(1),
    "readmitted_30_days": np.random.choice([0, 1], N, p=[0.82, 0.18]),
    "mortality_flag": np.random.choice([0, 1], N, p=[0.94, 0.06]),
    "ed_admission_flag": np.random.choice([0, 1], N, p=[0.55, 0.45]),
    "estimated_cost": np.random.normal(loc=8500, scale=3000, size=N).round(2)
})

df["length_of_stay"] = df["length_of_stay"].clip(lower=1)
df["estimated_cost"] = df["estimated_cost"].clip(lower=1000)

df.to_csv(
    "data/synthetic/hospital_operations_synthetic.csv",
    index=False
)

print("Synthetic hospital dataset created successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")