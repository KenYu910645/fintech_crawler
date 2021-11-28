from sklearn import tree
import pandas as pd 
import numpy as np
import os 
import math
from sklearn.neural_network import MLPClassifier, MLPRegressor

OUTPUT_FILE = "predict.csv"
train_df = pd.read_csv('train_clean_no_null.csv')
test_df  = pd.read_csv('test_clean_no_null.csv')
N_TRAIN_DATA = train_df.shape[0]
N_TEST_DATA = test_df.shape[0]
N_LABEL_IN_TRAIN = train_df[(train_df.label == 1)].shape[0]
IGNORE_LIST = []# ["have_credit_card", "have_housing_loan", "have_personal_loan", "previous_connect_weekday"]
print(f"Number of training data : {N_TRAIN_DATA}")
print(f"Number of label=1 in training data : {N_LABEL_IN_TRAIN}")
print(f"Number of testing data : {N_TEST_DATA}")

## Preprocess training and testing data
for df in [train_df, test_df]:
    for col_name in df:
        # if (col_name == 'marital'):
        #     for i, val in enumerate(df[col_name]):
        #         if df.at[i, col_name] == "unknown":
        #             df.at[i, col_name] = np.nan # float('nan')# "NULL"# "NULL"
        
        # Binarization 
        if col_name == 'after_campaign_connect_day': # TODO encode into one-hot vector?
            for i, val in enumerate(df[col_name]):
                if val > -1:
                    df.at[i, col_name] = 1
                else:
                    df.at[i, col_name] = -1

Y = train_df["label"]
X = train_df.drop(['label', 'index'] + IGNORE_LIST, axis=1)
X = pd.get_dummies(X)

X_test = test_df.drop(['index'] + IGNORE_LIST, axis=1)
X_test = pd.get_dummies(X_test)
# print(X)
# print(Y)
for i in range(Y.shape[0]):
    if (Y.loc[i] == 1):
        Y.loc[i] = 10

#######################
#### Decision Trees ###
#######################
# print("Training decision tree")
# dt = tree.DecisionTreeClassifier()
# dt = dt.fit(X, Y)
# print("Predicting")
# pred = dt.predict(X_test)

###########
### MLP ###
###########
# N_ITER = 100
# print("Training MLPClassifier")
# clf = MLPClassifier(hidden_layer_sizes = (33, 25,20, 15, 10, 5), verbose = True, max_iter = N_ITER, random_state=1, tol=1e-5, n_iter_no_change=N_ITER)# (solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X, Y)
# print("Predicting")
# pred = clf.predict(X_test)
# print(pred)

######################
### MLP Regression ###
######################
N_ITER = 3000
print("Training MLPClassifier")
clf = MLPRegressor(hidden_layer_sizes = (33, 25, 20, 15, 10, 5), verbose = True, max_iter = N_ITER, random_state=1, tol=1e-5, n_iter_no_change=N_ITER)# (solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, Y)
print("Predicting")
pred = clf.predict(X_test)
print(pred)


####################
### RandomForest ###
####################
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, random_state=0, verbose=1, class_weight={0:1, 10:10})
# print("Training RandomForestClassifier")
# clf.fit(X, Y)
# print("Predicting")
# # pred = clf.predict(X_test)
# pred = clf.predict_proba(X_test)
# print(pred.shape)
# 

THRESHOLD = 1 # 0.5
pred_true = 0
# n_no_friend = 0
# n_null = 0
s = "index,label\n"
for i in range(pred.shape[0]):
    if pred[i] > THRESHOLD:
        s += f"{i+20000},1\n"
        pred_true += 1
    else:
        s += f"{i+20000},0\n"
print(f"Number of predict true: {pred_true}")

with open(OUTPUT_FILE, "w") as f:
    f.write(s)
