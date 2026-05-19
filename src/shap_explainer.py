import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    "data/synthetic/hospital_operations_synthetic.csv"
)

# Load model
model = joblib.load(
    "src/models/readmission_model.pkl"
)

# Load encoders
encoders = joblib.load(
    "src/models/label_encoders.pkl"
)

# Encode categorical variables
categorical_cols = [
    "department",
    "payer_type",
    "primary_diagnosis"
]

for col in categorical_cols:
    df[col] = encoders[col].transform(df[col])

# Features
X = df[
    [
        "department",
        "payer_type",
        "primary_diagnosis",
        "length_of_stay",
        "estimated_cost",
        "mortality_flag",
        "ed_admission_flag"
    ]
]

# SHAP explainer
explainer = shap.Explainer(model)

shap_values = explainer(X)

# Summary plot
# Summary plot
plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "assets/images/shap_readmission_summary.png",
    dpi=300,
    bbox_inches="tight"
)

print("SHAP summary plot saved successfully.")