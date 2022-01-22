import pandas as pd
import time
# 2, 6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48

try:
    t_start = time.time()
    with pd.read_csv('tbrain_cc_training_48tags_hash_final.csv', chunksize = 10 ** 6) as reader:
        for i, df in enumerate(reader):
            bool_df = df[   (df.shop_tag != '2') &\
                            (df.shop_tag != '6') &\
                            (df.shop_tag != '10') &\
                            (df.shop_tag != '12') &\
                            (df.shop_tag != '13') &\
                            (df.shop_tag != '15') &\
                            (df.shop_tag != '18') &\
                            (df.shop_tag != '19') &\
                            (df.shop_tag != '21') &\
                            (df.shop_tag != '22') &\
                            (df.shop_tag != '25') &\
                            (df.shop_tag != '26') &\
                            (df.shop_tag != '36') &\
                            (df.shop_tag != '37') &\
                            (df.shop_tag != '39') &\
                            (df.shop_tag != '48')]
            
            df = df.drop(bool_df.index)
            df.to_csv(f'batch_{int(i/10)}_{int(i%10)}.csv', index=False)
            print(f"Saved batch_{i} to batch_{int(i/10)}_{int(i%10)}.csv.")

    print(f"Finished data split. takes {time.time() - t_start} sec.")
except FileNotFoundError as e:
    print(e)
    print("Can't find training data. Please use ./get_train_data.sh to download the training data.") 