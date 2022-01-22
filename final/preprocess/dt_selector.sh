cat train.csv | grep -E '^dt|^18' > train_last_six.csv
cat train.csv | grep '^19' >> train_last_six.csv
cat train.csv | grep '^20' >> train_last_six.csv
cat train.csv | grep '^21' >> train_last_six.csv
cat train.csv | grep '^22' >> train_last_six.csv
cat train.csv | grep '^23' >> train_last_six.csv
cat train.csv | grep '^24' >> train_last_six.csv