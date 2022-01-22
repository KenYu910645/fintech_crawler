import pandas as pd

# df_output.to_csv(f'train_merged.csv', index=False)
df = pd.read_pickle('train_merged.pkl')
# df = pd.read_csv('train_merged.csv')
# df.to_pickle("train_merged.pkl")
example_df = pd.read_csv('example.csv')
example_df = example_df.sort_values(by=['chid'])
print(example_df)

top1 = []
top2 = []
top3 = []
for chid in range(10000000, 10500000):
    row = df.loc[ df['chid'] == chid ]
    if row.empty:
        top1.append(15)
        top2.append(48)
        top3.append(2)
    else:
        top1.append(row['shop_tag_1'].values[0])
        top2.append(row['shop_tag_2'].values[0])
        top3.append(row['shop_tag_3'].values[0])
    
    if chid%10000 == 0:
        print(chid)

example_df = example_df.drop(['top1', 'top2', 'top3'],  axis=1)

example_df['top1'] = top1
example_df['top2'] = top2
example_df['top3'] = top3

print(example_df)
example_df.to_csv(f'base_line.csv', index=False)