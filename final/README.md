# fintech_final

## Usage

Download original training data
```
bash get_original_train_data.sh
```

Download cleaned training data, only shop_tag==2,6,10,12,13,15,18,19,21,22,25,26,36,37,39,48 datas are included in this .csv

```
bash get_train_clean_data.sh
```

Try to load train.csv and print statistic result
```
python exploer.py
```

Download preprocessed training data.
In this csv file, credit card details are omitted, only spent money and categories are presented 
Spanning duration is three months, which means I calculate top3 for every three months.

```
bash get_train_processed_data.sh
```
