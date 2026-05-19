import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
df = pd.read_csv(
    "data/synthetic/hospital_operations_synthetic.csv"
)

# Convert dates
df["admission_date"] = pd.to_datetime(df["admission_date"])

# Monthly admissions
monthly_admissions = (
    df.groupby(
        pd.Grouper(
            key="admission_date",
            freq="ME"
        )
    )
    .size()
    .reset_index(name="admissions")
)

# Time index
monthly_admissions["time_index"] = np.arange(len(monthly_admissions))

# Features and target
X = monthly_admissions[["time_index"]]
y = monthly_admissions["admissions"]

# Train forecasting model
model = LinearRegression()
model.fit(X, y)

# Forecast next 6 months
future_index = np.arange(
    len(monthly_admissions),
    len(monthly_admissions) + 6
).reshape(-1, 1)

future_predictions = model.predict(future_index)

# Future dates
future_dates = pd.date_range(
    monthly_admissions["admission_date"].max(),
    periods=7,
    freq="ME"
)[1:]

forecast_df = pd.DataFrame({
    "admission_date": future_dates,
    "forecasted_admissions": future_predictions
})

# Plot
plt.figure(figsize=(12,6))

plt.plot(
    monthly_admissions["admission_date"],
    monthly_admissions["admissions"],
    label="Historical Admissions"
)

plt.plot(
    forecast_df["admission_date"],
    forecast_df["forecasted_admissions"],
    linestyle="--",
    label="Forecasted Admissions"
)

plt.title("Hospital Admissions Forecast")
plt.xlabel("Date")
plt.ylabel("Admissions")
plt.legend()

plt.tight_layout()

plt.savefig(
    "assets/images/admissions_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

print("Admissions forecast saved successfully.")