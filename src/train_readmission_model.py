import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv(
    "data/synthetic/hospital_operations_synthetic.csv"
)

# Encode categorical variables
categorical_cols = [
    "department",
    "payer_type",
    "primary_diagnosis"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

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

# Target
y = df["readmitted_30_days"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(
    model,
    "src/models/readmission_model.pkl"
)

# Save encoders
joblib.dump(
    label_encoders,
    "src/models/label_encoders.pkl"
)

print("\nModel saved successfully.")