import pandas as pd

# df = pd.read_csv('train_last_six.csv')
df = pd.read_csv('train.csv')
# df = pd.read_pickle('train_sort_by_chid_six_month.pkl')

df = df.sort_values(by=['chid'])
# df.to_pickle("./train_sort_by_chid_all.pkl")

df.to_csv('train_sort_by_chid_all_month.csv', index=False)
