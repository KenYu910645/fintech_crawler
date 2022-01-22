import pandas as pd

# load train.csv
D = {} # defaultdict(dict)
N_DATA = 0

output_df = None 
try:
    with pd.read_csv('train.csv', chunksize = 10 ** 6) as reader:
        for chunk_idx, df in enumerate(reader):
            N_DATA += df.shape[0]
            print(f"Number of rows = {N_DATA}")
            
            try:
                output_df = output_df.append(df.loc[df['dt'] >= 18])
            except AttributeError as e :
                output_df = df.loc[df['dt'] >= 18]

            print(f"Loaded chunk {chunk_idx}")

except FileNotFoundError as e:
    print(e)
    print("Can't find train data. Please use ./get_train_data.sh to download the training data.") 

output_df.to_csv("train_last_six.csv", index=False)
