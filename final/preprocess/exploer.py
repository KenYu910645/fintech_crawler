import pandas as pd
import pprint
import math
from collections import defaultdict
import os

# load train.csv
d_all = {}
N_DATA = 0
try:
    with pd.read_csv('train.csv', chunksize = 10 ** 6) as reader:
        for chunk_idx, df in enumerate(reader):
            N_DATA += df.shape[0]
            print(f"Number of rows = {N_DATA}")
            # Init d_all
            if d_all == {}:
                for col_name in df.columns:
                    d_all[col_name] = defaultdict(int)
            for col_name in df:
                for i, val in enumerate(df[col_name]):
                    if type(val) == float and math.isnan(val):
                            d_all[col_name]["NULL"] += 1
                    else:
                        d_all[col_name][val] += 1
            print(f"Loaded chunk {chunk_idx}")

except FileNotFoundError as e:
    print(e)
    print("Can't find train data. Please use ./get_train_data.sh to download the training data.") 

# Print message
print(f"Number of custumers = {len(d_all['chid'])}")

for col_name in d_all:
    if col_name in ['dt', 'shop_tag', 'masts', 'educd', 'trdtp', 'naty', 'poscd', 'cuorg', 'gender_code', 'primary_card']:
        s = ""
        for i in d_all[col_name]:
            s += f"{i} : {d_all[col_name][i]} ({round(100*d_all[col_name][i]/N_DATA, 2)}%)\n"
        print(f"============ {col_name} ============")
        print(s)

