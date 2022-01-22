from collections import defaultdict
import pandas as pd
import time
import pickle
import math 
N_DATA = 0
MONTH_RANGE = 3 # Months

# Calculate duration list
DURA_LIST = [] # Duration list
for i in range(int(24/MONTH_RANGE)):
    m_start = i*MONTH_RANGE + 1
    m_end = m_start + MONTH_RANGE -1 
    DURA_LIST.append(f"{m_start}to{m_end}")
print(f"DURA_LIST = {DURA_LIST}")

# Load dataframe
# df = pd.read_pickle('train_sort_by_chid_six_month.pkl')
df = pd.read_pickle('train_sort_by_chid_all_month.pkl')
print ("Finish reading pickle")

# Drop useless column
drop_column = ['txn_cnt', 'domestic_offline_cnt', 'domestic_online_cnt',
               'overseas_offline_cnt','overseas_online_cnt', 'domestic_offline_cnt', 'domestic_online_cnt',
               'overseas_offline_amt_pct', 'overseas_online_amt_pct','domestic_offline_amt_pct','domestic_online_amt_pct',
               'card_other_txn_cnt', 'card_other_txn_amt_pct']
for i in range(1, 15):
    drop_column.append(f"card_{i}_txn_cnt")
    drop_column.append(f"card_{i}_txn_amt_pct")
df = df.drop(drop_column, axis=1)
print(df)

# Drop duplicate row 
df_output = df.drop_duplicates(subset ="chid")
df_output = df_output.drop(['dt', 'shop_tag', 'txn_amt'],  axis=1)
print(df_output)

# Init T_FEATURE
T_FEATURE = {}
for rank in range(1, 3+1):
    for dura in DURA_LIST:
        T_FEATURE[f"shop_tag_{dura}_{rank}"] = []
        T_FEATURE[f"shop_tag_txn_{dura}_{rank}"] = []

# Group chid in df
g_chid = df.groupby('chid')
print("Finish grouping")

t_start = time.time()
for chid in g_chid.groups:

    # Init ans
    ans = {}
    for dura in DURA_LIST:
        ans[dura] = defaultdict(int)

    # Calculate ans
    for index, row in g_chid.get_group(chid).iterrows():
        dura = DURA_LIST[math.floor((row['dt']-1) / MONTH_RANGE)]
        ans[dura][row['shop_tag']] += row['txn_amt']


    # Sort ans
    for dura in DURA_LIST:
        ans[dura] = sorted(ans[dura].items(), key=lambda x:x[1], reverse=True)

    # Fill in T_FEATURE
    for rank in range(1, 3+1):
        for dura in DURA_LIST:
            if rank-1 < len(ans[dura]):
                T_FEATURE[f'shop_tag_{dura}_{rank}'].append(ans[dura][rank-1][0])
                T_FEATURE[f'shop_tag_txn_{dura}_{rank}'].append(ans[dura][rank-1][1])
            else:
                T_FEATURE[f'shop_tag_{dura}_{rank}'].append(0)
                T_FEATURE[f'shop_tag_txn_{dura}_{rank}'].append(0)

    # Output debug message
    if (chid % 1000 == 0):
        print(chid)
        print(f"Takes {time.time() - t_start} seconds.")
        t_start = time.time()

# Save pickle
with open("T_FEATURE_ALLL.pkl", "wb") as tf:
    pickle.dump(T_FEATURE, tf)
print(f"Saved T_FEATURE_ALLL to T_FEATURE_ALLL.pkl")

# Load pickle # This is used if programm intereputed.
# with open("T_FEATURE.pkl", "wb") as tf:
#     T_FEATURE = pickle.load(tf)

# Append new features into df_output
for dura in DURA_LIST:
    for rank in range(1, 3+1):
        df_output[f'shop_tag_{dura}_{rank}']     = T_FEATURE[f'shop_tag_{dura}_{rank}']
        df_output[f'shop_tag_txn_{dura}_{rank}'] = T_FEATURE[f'shop_tag_txn_{dura}_{rank}']
print(df_output)

# Output result to csv
df_output.to_csv(f'feature_selection_with_dt_all.csv', index=False)
print(f"Wrote result to feature_selection_with_dt_all.csv")
