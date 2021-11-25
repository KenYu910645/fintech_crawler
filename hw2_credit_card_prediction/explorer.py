import pandas as pd
import pprint
import math

# load train.csv
df = pd.read_csv('train_clean.csv')
from collections import defaultdict
import os 

N_DATA = df.shape[0]
output_file = "result.csv"
output_label_file = "result_label.csv"
if os.path.exists(output_file):
    os.remove(output_file)
if os.path.exists(output_label_file):
    os.remove(output_label_file)

with open(output_file, "w") as f:
    f.write("Total, \n")
with open(output_label_file, "w") as f:
    f.write("label, \n")


def round_to_point_5(x):
    return round(x * 2) / 2

def round_to_5(x, base=5):
    try:
        return base * round(x/base)
    except ValueError as e:
        return x

############  check data   ###############
for col_name in df:
    if col_name == "index":
        continue
    # Round age to nearest 5
    elif (col_name == 'age'):
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
    
    d = defaultdict(int)
    for i, val in enumerate(df[col_name]):
        if type(val) == float and math.isnan(val):
                d["NULL"] += 1
        else:
            d[val] += 1
    
    
    s = ""
    for i in d:
        s += f"{i} : {d[i]} ({round(100*d[i]/N_DATA, 2)}%)\n"
    print(f"============ {col_name} ============")
    print(s)
    
    with open(output_file, "a") as f:
        f.write(f"{col_name}, \n")
        f.write(s)
        # for i in d:
        #     f.write(f"{i}, {d[i]} ({round(100*d[i]/N_DATA, 2)}%)\n")


# I think NULL value is not very common, so it's not very important for now
print("**************************************************************************")
label_df = df[(df.label == 1)] # & (df.marital == "married")]
for column_name in label_df:
    if column_name == "index":
        continue
    d = defaultdict(int)
    for i, val in enumerate(label_df[column_name]):
        if type(val) == float:
            if math.isnan(val):
                d["NULL"] += 1
            else:
                d[round(val)] += 1
        else:
            d[val] += 1
    
    print(f"============ {column_name} ============")
    for i in d:
        print(f"{i} : {d[i]} ({round(100*d[i]/label_df.shape[0], 2)}%)")
    
    with open(output_label_file, "a") as f:
        f.write(f"{column_name}, \n")
        for i in d:
            f.write(f"{i}, {d[i]} ({round(100*d[i]/label_df.shape[0], 2)}%)\n")




