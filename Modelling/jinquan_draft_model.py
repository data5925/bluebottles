# Binary LSTM for Bluebottle Presence
# Target: `presence` (binary classification)
# Features: all other columns(without `bluebottles`)
# LSTM with class weights to address imbalance
# TensorFlow 
# Initial attempt (v1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv("cleaned_data.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(by='time')
df.set_index('time', inplace=True)

df = df.drop(columns=['bluebottles'])

target_col = 'presence'
features = df.drop(columns=[target_col])
target = df[[target_col]]

scaler_x = MinMaxScaler()
scaled_x = scaler_x.fit_transform(features)
def create_sequences(x, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(x) - time_steps):
        Xs.append(x[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X, y = create_sequences(scaled_x, target.values, time_steps)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train.flatten()),
    y=y_train.flatten()
)
class_weights_dict = dict(enumerate(class_weights_array))
print("Computed class weights:", class_weights_dict)

model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weights_dict,
    verbose=1
)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.7).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
plt.figure(figsize=(12, 5))
plt.plot(y_test[:300], label='Actual', linewidth=1)
plt.plot(y_pred[:300], label='Predicted', linewidth=1)
plt.title("LSTM Binary Classification - presence (with class_weight, first 300 test samples)")
plt.xlabel("Time Step")
plt.ylabel("Presence (0 or 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Optimized with stacked LSTM, thershold, epoch, (dropout),more units, early stopping, longer visualization
#version2 3 4 5 6 7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("final_data.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(by='time')
df.set_index('time', inplace=True)
df = df.drop(columns=['bluebottles'])  # Drop unused column


target_col = 'presence'
features = df.drop(columns=[target_col])
target = df[[target_col]]

scaler_x = MinMaxScaler()
scaled_x = scaler_x.fit_transform(features)


def create_sequences(x, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(x) - time_steps):
        Xs.append(x[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X, y = create_sequences(scaled_x, target.values, time_steps)


split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train.flatten()),
    y=y_train.flatten()
)
class_weights_dict = dict(enumerate(class_weights_array))
print("Computed class weights:", class_weights_dict)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int) 

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

plt.figure(figsize=(14, 5))
plt.plot(y_test[:500], label='Actual', linewidth=1)
plt.plot(y_pred[:500], label='Predicted', linewidth=1)
plt.title("LSTM Binary Classification - presence (Stacked LSTM, first 500 test samples)")
plt.xlabel("Time Step")
plt.ylabel("Presence (0 or 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
