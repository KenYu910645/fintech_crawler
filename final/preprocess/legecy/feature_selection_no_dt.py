import pandas as pd
N_DATA = 0

df = pd.read_pickle('train_sort_by_chid_six_month.pkl')
print ("Finish reading pickle")

# Drop useless column
drop_column = ['dt', 'txn_cnt', 'domestic_offline_cnt', 'domestic_online_cnt',
               'overseas_offline_cnt','overseas_online_cnt', 'domestic_offline_cnt', 'domestic_online_cnt',
               'overseas_offline_amt_pct', 'overseas_online_amt_pct','domestic_offline_amt_pct', 'domestic_online_amt_pct',
               'card_other_txn_cnt', 'card_other_txn_amt_pct']
for i in range(1, 15):
    drop_column.append(f"card_{i}_txn_cnt")
    drop_column.append(f"card_{i}_txn_amt_pct")
df = df.drop(drop_column, axis=1)
print(df)

# Drop duplicate row 
df_output = df.drop_duplicates(subset ="chid")
df_output = df_output.drop(['shop_tag', 'txn_amt'],  axis=1)

print(df_output)

g_chid = df.groupby('chid')
print("Finish grouping")

shop_tag_1 = []
shop_tag_txn_1 = []
shop_tag_2 = []
shop_tag_txn_2 = []
shop_tag_3 = []
shop_tag_txn_3 = []

tmp_idx = 0
for chid in g_chid.groups:
    ans = {2: 0,
           6: 0,
           10: 0,
           12: 0, 
           13: 0, 
           15: 0, 
           18: 0, 
           19: 0, 
           21: 0, 
           22: 0, 
           25: 0, 
           26: 0, 
           36: 0, 
           37: 0, 
           39: 0, 
           48: 0} # [[shop_tag, v]]
    
    g_shop_tag = g_chid.get_group(chid).groupby('shop_tag')
    for shop_tag in g_shop_tag.groups:
        ans[shop_tag] += g_shop_tag.get_group(shop_tag)['txn_amt'].sum()

    #     for chid_in_group in g_chid.get_group(chid).iterrows(): # TODO this might be bad 
    #         # ans[row['shop_tag']] += row['txn_amt']
    #         g_chid.get_group(chid).chid
    #         ans[row['shop_tag']]

    ans = sorted(ans.items(), key=lambda x:x[1], reverse=True)

    try:
        shop_tag_1.append(ans[0][0])
        shop_tag_txn_1.append(ans[0][1])
    except KeyError as e :
        shop_tag_1.append(0)
        shop_tag_txn_1.append(0)

    try:
        shop_tag_2.append(ans[1][0])
        shop_tag_txn_2.append(ans[1][1])
    except KeyError as e :
        shop_tag_2.append(0)
        shop_tag_txn_2.append(0)

    try:
        shop_tag_3.append(ans[2][0])
        shop_tag_txn_3.append(ans[2][1])
    except KeyError as e :
        shop_tag_3.append(0)
        shop_tag_txn_3.append(0)

    if (tmp_idx % 1000 == 0):
        print(chid)
    tmp_idx += 1

df_output['shop_tag_1'] = shop_tag_1
df_output['shop_tag_txn_1'] = shop_tag_txn_1
df_output['shop_tag_2'] = shop_tag_2
df_output['shop_tag_txn_2'] = shop_tag_txn_2
df_output['shop_tag_3'] = shop_tag_3
df_output['shop_tag_txn_3'] = shop_tag_txn_3
print(df_output)

df_output.to_csv(f'train_merged.csv', index=False)
