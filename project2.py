import os
import pandas as pd

def load_credit_data():
    csv_path = os.path.join("./datasets/dataset_31_credit-g.csv")
    return pd.read_csv(csv_path)

credit_data = load_credit_data()
print(credit_data.head())

## Data Analysis
