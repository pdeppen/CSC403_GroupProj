import os
import pandas as pd
import matplotlib.pyplot as plt

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

print(credit_data.info())
print(credit_data.describe())