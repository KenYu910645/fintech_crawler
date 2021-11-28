from collections import defaultdict
import pandas as pd
import numpy as np

OUTPUT_FILE = "train_clean_no_null.csv"
train_df = pd.read_csv('train_clean.csv')

change = []
n_null = 0

for i in range(train_df.shape[0]):
    x = train_df.loc[i, :]
    for col_idx in range(1, train_df.shape[1] ):
        col_name = x.index[col_idx]
        if (type(x[col_name]) == np.float64 or type(x[col_name]) == float) and np.isnan(x[col_name]):
            n_null += 1

            # Find similiar data
            train_df_tmp = train_df.drop([col_name], axis=1)
            x_tmp = x.drop([col_name])

            mask = None
            for f in ['age', 'euducation_level', 'job', 'marital', 'connect_method', 'last_campaign_outcomes']:
                if f == col_name: # ignore missing feature
                    continue
                else:
                    try:
                        mask = mask & (train_df_tmp.loc[:, f] == x[f])
                    except TypeError as e :
                        mask = (train_df_tmp.loc[:, f] == x[f])

            same_feature_df = train_df_tmp[mask]

            # Find maximum likelihood in same_feature
            dic = defaultdict(int)
            for row_idx in same_feature_df.index:
                dic[train_df.loc[row_idx, col_name]] += 1
            
            max_likeli = 0
            ans = None
            for d in dic:
                if (type(d) == np.float64 or type(d) == float) and np.isnan(d):
                    continue
                else:
                    if dic[d] > max_likeli:
                        max_likeli = dic[d]
                        ans = d

            if (type(ans) == np.float64 or type(ans) == float) and np.isnan(ans):
                print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG still null ")
            if ans == None:
                print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG can't find ans ")
            print(f"In {i} row, change {col_name} {train_df.loc[i, col_name]} to {ans}")
            change.append( (i, col_name, ans) )

# Write to new csv 
for r_idx, col_name, new_val in change:
    train_df.loc[r_idx, col_name] = new_val
train_df.to_csv(OUTPUT_FILE, index=False)

print(f"Total {n_null} Null has been changed")
