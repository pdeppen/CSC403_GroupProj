import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

def load_credit_data():
    csv_path = os.path.join("./datasets/dataset_31_credit-g.csv")
    return pd.read_csv(csv_path)

# load data
credit_data = load_credit_data()
# apply label encoder
credit_data = credit_data.apply(LabelEncoder().fit_transform)

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

# convert to array
credit_data = credit_data.values

# data set
credit_data_data = credit_data

# remove target column from data set
credit_data_data = np.delete(credit_data_data, 9, axis=1)

# target set
credit_data_target = credit_data
credit_data_target = np.delete(credit_data_target, np.s_[0:9], axis=1)
credit_data_target = credit_data_target.ravel()


# split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(credit_data_data, credit_data_target, test_size=0.5)

shuffle_index = np.random.permutation(500)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train_yes = (y_train == 1)
y_test_yes = (y_test == 1)

sgd_clf = SGDClassifier(max_iter=20, tol=np.infty, random_state=42)
sgd_clf.fit(x_train, y_train_yes)

print(cross_val_score(sgd_clf, x_train, y_train_yes, cv=3, scoring="accuracy"))

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_yes, cv=3)

print(confusion_matrix(y_train_yes, y_train_pred))

