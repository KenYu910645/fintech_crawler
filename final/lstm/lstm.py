import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.DataFrame([1,2,3,4,5,6])
print(df)
# Data normalization
scaler = MinMaxScaler(feature_range = (0, 10))
apple_training_scaled = scaler.fit_transform(df)

print(apple_training_scaled)



features_set = apple_training_scaled[:5] # This is input feature, 0~23 month
labels = apple_training_scaled[5] # This is output label, 24 month
# for i in range(60, 1260):
#     features_set.append(apple_training_scaled[i-60:i, 0])
#     labels.append(apple_training_scaled[i, 0])

print(f"features_set = {features_set}")
print(f"labels = {labels}")


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Creating LSTM and Dropout Layers
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
# Creating Dense Layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Start training
model.fit(features_set, labels, epochs = 100, batch_size = 32)