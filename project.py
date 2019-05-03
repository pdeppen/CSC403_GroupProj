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
# credit_data[:, 0] = labelencoder.fit_transform(credit_data[:, 0])
# credit_data.apply(LabelEncoder().fit_transform)

credit_data = credit_data.apply(LabelEncoder().fit_transform)
# labelencoder.fit(credit_data["class"])
# print(labelencoder.classes_)
# credit_data["class"] = labelencoder.transform(credit_data["class"])
# print(credit_data["class"])

# print(credit_data)
# print(credit_data["personal_status"])

# labelencoder.fit


corr_matrix = credit_data.corr()
print(corr_matrix["class"].sort_values(ascending=False))


attributes = ["duration", "age", "credit_amount"]
scatter_matrix(credit_data[attributes], figsize=(12, 8))
plt.show()

from pandas.plotting import scatter_matrix

attributes = ["class", "checking_status", "savings_status", "age"]

# scatter_matrix(credit_data[attributes], figsize=(12,8))
# plt.show()
>>>>>>> master
