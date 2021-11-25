from collections import defaultdict
import os 
import pandas as pd
import math
import numpy as np
OUTPUT_FILE = "predict.csv"

# load train.csv
THESHOLD = 0.1
train_df = pd.read_csv('train_clean.csv')
test_df  = pd.read_csv('test_clean.csv')
N_TRAIN_DATA = train_df.shape[0]
N_TEST_DATA = test_df.shape[0]
N_LABEL_IN_TRAIN = train_df[(train_df.label == 1)].shape[0]
IGNORE_LIST = ["have_credit_card", "have_housing_loan", "have_personal_loan", "previous_connect_weekday"]
print(f"Number of training data : {N_TRAIN_DATA}")
print(f"Number of label=1 in training data : {N_LABEL_IN_TRAIN}")
print(f"Number of testing data : {N_TEST_DATA}")

def round_to_point_5(x):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""
    return round(x * 2) / 2

def round_to_5(x, base=5):
    try:
        return base * round(x/base)
    except ValueError as e:
        return x

## Preprocess training and testing data
for df in [train_df, test_df]:
    for col_name in df:
        # Round age to nearest 5
        if (col_name == 'age'):
            for i, val in enumerate(df[col_name]):
                df.at[i, col_name] = round_to_5(val)
        elif (col_name == 'marital'):
            for i, val in enumerate(df[col_name]):
                if df.at[i, col_name] == "unknown":
                    df.at[i, col_name] = np.nan # float('nan')# "NULL"# "NULL"
        
        elif (col_name == "employment_rate" or \
              col_name == "consumer_price_index" or \
              col_name == "consumer_confidence_index"):
            for i, val in enumerate(df[col_name]):
                df.at[i, col_name] = round_to_point_5(val)
        
        # Ignore these features
        elif (col_name in IGNORE_LIST):
            for i, val in enumerate(df[col_name]):
                df.at[i, col_name] = "NULL"

# print(train_df)
# print(test_df)


pred_true = 0
n_no_friend = 0
n_null = 0
s = "index,label\n"
P_cache = defaultdict(dict) # { 'age':{'30.0': (0.1, 0,2) }  } # (P_x, P_xly)
P_y = 0.1509

for i in range(test_df.shape[0]):
    x = test_df.loc[i, :]
    posterior_acc = 1
    for col_idx in range(1, test_df.shape[1] ):
        col_name = x.index[col_idx]
        
        # Deal with special cases
        if col_name == "age" and x[col_name] == 15:
            x[col_name] = 20
        if col_name == "campaign_connect_times" and x[col_name] == 16:
            x[col_name] = 17
        if col_name == "after_campaign_connect_day" and x[col_name] in [21, 26, 30]:
            x[col_name] = 27

        # Check data in cache
        if x[col_name] in P_cache[col_name]:
            P_x, P_xly = P_cache[col_name][ x[col_name] ]
        else:
            same_feature_df = train_df[(train_df.loc[:, col_name] == x[col_name])]
            same_feature_df_label = same_feature_df[(same_feature_df.label == 1)]
            P_x = same_feature_df.shape[0] / N_TRAIN_DATA
            P_xly = same_feature_df_label.shape[0] / N_LABEL_IN_TRAIN
            # Store result in cache
            P_cache[col_name][ x[col_name] ] = (P_x, P_xly)
        
        # Null data in testset
        if (type(x[col_name]) == np.float64 or type(x[col_name]) == float)and np.isnan(x[col_name]):
            P_y_given_x = P_y
            n_null += 1

        # Special cases
        elif col_name == "campaign_connect_times" and  x[col_name] > 17: 
            P_y_given_x = 0 # This data is doomed

        elif P_x == 0 or P_xly == 0:
            print(f"can't find {x[col_name]} in column: {col_name}")
            P_y_given_x = P_y
            n_no_friend += 1
        else:
            P_y_given_x = (P_xly)/P_x
        posterior_acc *= P_y_given_x
        # print(f"P(x) = {P_x}, P(x|y) = {P_x_given_label_1}, P(y|x) = {P_y_given_x}")

    # print(f"{x[0]}: posterior_acc = {posterior_acc}")

    if posterior_acc > THESHOLD:
        pred = 1
        pred_true += 1
    else:
        pred = 0
    s += f"{x[0]},{pred}\n"

print(f"THRESHOLD = {THESHOLD}")
print(f"Number of predict ture = {pred_true}")
print(f"Number of no friends = {n_no_friend}")
print(f"Number of NULL = {n_null}")
with open(OUTPUT_FILE, "w") as f:
    f.write(s)


''' Special cases in testing data
can't find 15.0 in column: age
can't find 30.0 in column: after_campaign_connect_day
can't find 21.0 in column: after_campaign_connect_day
can't find 26.0 in column: after_campaign_connect_day
can't find 21.0 in column: after_campaign_connect_day
can't find 42.0 in column: campaign_connect_times
can't find 15.0 in column: age
can't find 21.0 in column: after_campaign_connect_day
can't find 42.0 in column: campaign_connect_times
can't find 15.0 in column: age
can't find 15.0 in column: age
can't find 41.0 in column: campaign_connect_times
'''