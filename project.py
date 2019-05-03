import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def load_credit_data():
    csv_path = os.path.join("./datasets/dataset_31_credit-g.csv")
    return pd.read_csv(csv_path)

credit_data = load_credit_data()

# prints 5 rows of the data set
# print(credit_data.head())

# description of the data
# print(credit_data.info())

# to show all categorical values using the following for a specific feature
# print(credit_data["housing"])

# HISTOGRAMS FOR NUMERICAL DATA
# plt.title("Age Histogram")
# credit_data["age"].hist(bins=50, figsize=(20,15))
# plt.title("Credit Amount Histogram")
# credit_data["credit_amount"].hist(bins=50, figsize=(20,15))
# plt.title("Duration Histogram")
# credit_data["duration"].hist(bins=50, figsize=(20,15))
# plt.show()

#

## Data Analysis

# print(credit_data.info())
# print(credit_data.describe())


#ENCODER
# credit_data_encoded, credit_data_categories = credit_data_cat.factorize()
# print(credit_data_encoded[:10])

# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# credit_data_1hot = encoder.fit_transform()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
# print(credit_data["personal_status"])

credit_data = credit_data.apply(LabelEncoder().fit_transform)


# print(credit_data)
# print(credit_data["personal_status"])

corr_matrix = credit_data.corr()
# print(corr_matrix["class"].sort_values(ascending=False))


attributes = ["duration", "age", "credit_amount"]
scatter_matrix(credit_data[attributes], figsize=(12, 8))

attributes = ["class", "checking_status", "savings_status", "age"]

# scatter_matrix(credit_data[attributes], figsize=(12,8))

# plt.show()

# dropping features
print(credit_data.info())
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

print(credit_data.info())
