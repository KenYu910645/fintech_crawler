from collections import defaultdict
import os 
import pandas as pd
OUTPUT_FILE = "predict.csv"

# load train.csv
train_df = pd.read_csv('train_clean.csv')
test_df  = pd.read_csv('test_clean.csv')
N_TRAIN_DATA = train_df.shape[0]
N_TEST_DATA = test_df.shape[0]
N_LABEL_IN_TRAIN = train_df[(train_df.label == 1)].shape[0]
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
                    df.at[i, col_name] = "NULL"
        
        elif (col_name == "employment_rate" or \
              col_name == "consumer_price_index" or \
              col_name == "consumer_confidence_index"):
            for i, val in enumerate(df[col_name]):
                df.at[i, col_name] = round_to_point_5(val)
        
        # Ignore these features
        elif (col_name == "have_credit_card" or \
              col_name == "have_housing_loan" or \
              col_name == "have_personal_loan" or \
              col_name == "previous_connect_weekday"):
            for i, val in enumerate(df[col_name]):
                df.at[i, col_name] = "NULL"

# print(train_df)
# print(test_df)

## TODO campaign connect time, after compaign connect day, before campaign connect time should use smooth
THESHOLD = 0.1
pred_true = 0
n_no_friend = 0
s = "index,label\n"
# 0.64364
for i in range(test_df.shape[0]):
    x = test_df.loc[i, :]
    # TODO NULL
    same_feature_df = train_df[(train_df.age == x.age) & \
                        (train_df.euducation_level == x.euducation_level) & \
                        (train_df.job == x.job) & \
                        (train_df.marital == x.marital) & \
                        (train_df.connect_method == x.connect_method)]
                        # (train_df.previous_connect_month == x.previous_connect_month) & \
                        # (train_df.campaign_connect_times == x.campaign_connect_times) & \
                        # (train_df.after_campaign_connect_day == x.after_campaign_connect_day)]
                        # (train_df.before_campaign_connect_times == x.before_campaign_connect_times) & \
                        # (train_df.last_campaign_outcomes == x.last_campaign_outcomes)]
                        # (train_df.employment_rate == x.employment_rate) & \
                        # (train_df.consumer_price_index == x.consumer_price_index) & \
                        # (train_df.consumer_confidence_index == x.consumer_confidence_index)]

    same_feature_df_label = same_feature_df[(same_feature_df.label == 1)]
    P_x = same_feature_df.shape[0] / N_TRAIN_DATA
    P_y = 0.1509
    P_x_given_label_1 = same_feature_df_label.shape[0] / N_LABEL_IN_TRAIN

    
    if P_x == 0: # TODO, not sure this is good
        P_y_given_x = 0
        n_no_friend += 1
    else:
        P_y_given_x = (P_y*P_x_given_label_1)/P_x
    
    # print(f"P(x) = {P_x}, P(x|y) = {P_x_given_label_1}, P(y|x) = {P_y_given_x}", )
    
    if P_y_given_x > THESHOLD:
        pred = 1
        pred_true += 1
    else:
        pred = 0
    s += f"{x[0]},{pred}\n"
print(f"THRESHOLD = {THESHOLD}")
print(f"Number of predict ture = {pred_true}")
print(f"Number of no friends = {n_no_friend}")

with open(OUTPUT_FILE, "w") as f:
    f.write(s)
