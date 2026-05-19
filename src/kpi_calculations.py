import pandas as pd


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def calculate_total_admissions(df):
    return len(df)


def calculate_average_los(df):
    return round(df["length_of_stay"].mean(), 2)


def calculate_readmission_rate(df):
    rate = df["readmitted_30_days"].mean() * 100
    return round(rate, 2)


def calculate_mortality_rate(df):
    rate = df["mortality_flag"].mean() * 100
    return round(rate, 2)


def calculate_average_cost(df):
    return round(df["estimated_cost"].mean(), 2)


def calculate_ed_admission_rate(df):
    rate = df["ed_admission_flag"].mean() * 100
    return round(rate, 2)