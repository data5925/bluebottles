# This model is based on the tutorial on GeeksforGeeks (https://www.geeksforgeeks.org/long-short-term-memory-lstm-rnn-in-tensorflow/)
# Note: no set seed so results are different every time (30 days window size is most consistent with accuracy > 0.9)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset
data = pd.read_csv('data/final_data.csv')
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)

# Separate features and target
features = data.drop(columns=['presence', 'bluebottles'])  
target = data['presence']

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Set window size (can try 7-14 days?)
window_size = 30
X, y, target_dates = [], [], []

# Create sequences (multivariate)
for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i])
    y.append(target[i])
    target_dates.append(data.index[i])

X = np.array(X)
y = np.array(y)
target_dates = np.array(target_dates)

# Train-test split (80/20)
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, target_dates, test_size=0.2, shuffle=False)

# Compute class weights for imbalance (https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (using class weights)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, class_weight=class_weights)

# Predict probabilities
predictions = model.predict(X_test)

# Convert probabilities to class labels (increase threshold to 0.7 for higher accuracy - fewer wrong predictions for presence, while not predicting absence all the time)
predicted_classes = (predictions >= 0.7).astype(int).flatten()

# Accuracy
accuracy = np.mean(predicted_classes == y_test)
print(f'Accuracy: {accuracy:.2f}')

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual Presence', alpha=0.5)
plt.plot(dates_test, predicted_classes, label='Predicted Presence', alpha=0.5)
plt.title('Actual vs Predicted Bluebottle Presence')
plt.xlabel('Date')
plt.ylabel('Presence (1 = Present, 0 = Absent)')
plt.legend()
plt.tight_layout()
plt.show()


