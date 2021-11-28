from collections import defaultdict
import os 
import pandas as pd
import math
import numpy as np
OUTPUT_FILE = "predict.csv"

train_df = pd.read_csv('train_clean_no_null.csv')
test_df  = pd.read_csv('test_clean_no_null.csv')
N_TRAIN_DATA = train_df.shape[0]
N_TEST_DATA = test_df.shape[0]
N_LABEL_IN_TRAIN = train_df[(train_df.label == 1)].shape[0]
IGNORE_LIST = ["age", "euducation_level", "marital", "have_credit_card", "have_housing_loan", "have_personal_loan", "previous_connect_weekday"]
print(f"Number of training data : {N_TRAIN_DATA}")
print(f"Number of label=1 in training data : {N_LABEL_IN_TRAIN}")
print(f"Number of testing data : {N_TEST_DATA}")


good = 0
bad = 0
s = "index,label\n"
N_RANDOM = 5000 - 1064
n_ran = N_RANDOM
for i in range(N_TEST_DATA):
    # if test_df.loc[i, "last_campaign_outcomes"] == "success": # 754, # 0.7261% in trianing data
    #     pass

    # if test_df.loc[i, "after_campaign_connect_day"] != -1: # 830 # 0.7089%
    #     pass # n += 1
    # 

    # I want this 
    if  test_df.loc[i, "last_campaign_outcomes"] == "success" or \
        test_df.loc[i, "after_campaign_connect_day"] != -1 or \
        test_df.loc[i, "before_campaign_connect_times"] > 2 or \
        test_df.loc[i, "age"] > 67 or \
        int(test_df.loc[i, "employment_rate"]) == 59 or \
        int(test_df.loc[i, "consumer_price_index"] ) == 95 or \
        int(test_df.loc[i, "consumer_confidence_index"] ) == -35:

        good += 1
        s += f"{i+20000},1\n"
    # I don't want this
    else:
        # test_df.loc[i, "previous_connect_month"] == "May" or \
        if  test_df.loc[i, "have_credit_card"] == "unknown" or \
            test_df.loc[i, "connect_method"] == "telephone"or \
            test_df.loc[i, "campaign_connect_times"] > 11 or \
            int(test_df.loc[i, "employment_rate"]) == 61 or \
            int(test_df.loc[i, "employment_rate"]) == 60 or \
            int(test_df.loc[i, "consumer_confidence_index"]) == -43 or \
            int(test_df.loc[i, "consumer_confidence_index"]) == -36:
            bad += 1
            s += f"{i+20000},0\n"
        else:
            # TODO, random select or ALL IN
            s += f"{i+20000},0\n"
            # if n_ran < N_RANDOM :
            #     s += f"{i+20000},1\n"
            #     n_ran += 1
            # else:
            #     s += f"{i+20000},0\n"

# for i in range(pred.shape[0]):
#     if pred[i] > THRESHOLD:
#         s += f"{i+20000},1\n"
#         pred_true += 1
#     else:
#         s += f"{i+20000},0\n"

print(f"good data = {good}")
print(f"bad data = {bad}")
with open(OUTPUT_FILE, "w") as f:
    f.write(s)


    

