import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def load_credit_data():
    csv_path = os.path.join("./datasets/dataset_31_credit-g.csv")
    return pd.read_csv(csv_path)

credit_data = load_credit_data()

# dropping features
credit_data = credit_data.drop("num_dependents", axis=1)
credit_data = credit_data.drop("purpose", axis=1)
credit_data = credit_data.drop("personal_status", axis=1)
credit_data = credit_data.drop("other_parties", axis=1)
credit_data = credit_data.drop("property_magnitude", axis=1)
credit_data = credit_data.drop("other_payment_plans", axis=1)
credit_data = credit_data.drop("existing_credits", axis=1)
credit_data = credit_data.drop("job", axis=1)
credit_data = credit_data.drop("own_telephone", axis=1)
credit_data = credit_data.drop("foreign_worker", axis=1)
credit_data = credit_data.drop("installment_commitment", axis=1)

