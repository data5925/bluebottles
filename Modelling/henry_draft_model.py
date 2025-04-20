import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
import tensorflow.keras.backend as K

# focal loss function (alpha = 0.75 produces best balance)
def binary_focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow((1 - p_t), gamma)
        return K.mean(-alpha_factor * modulating_factor * K.log(p_t))
    return focal_loss

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Load dataset 
data = pd.read_csv('data/final_data.csv')
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)
#data = data[data['beach.x'] == 0] # when uncommenting this line, remove beach.x in features and change threshold back to 0.5

# Separate features and target (add beach.x for features when including all beaches)
features = data[['beach.x','crt_salt','crt_u','crt_v','wnd_sfcWindspeed','wnd_uas','wnd_vas','wave_fp']]
target = data['presence']

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Create sequences (14 days is middleground)
window_size = 14
X, y, target_dates = [], [], []

for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i])
    y.append(target[i])
    target_dates.append(data.index[i])

X = np.array(X)
y = np.array(y)
target_dates = np.array(target_dates)

# Train-test split
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, target_dates, test_size=0.2, shuffle=False)

# SMOTE
X_train_flat = X_train.reshape((X_train.shape[0], -1))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)

# Reshape back to 3D for LSTM 
X_train = X_resampled.reshape((X_resampled.shape[0], window_size, X.shape[2]))
y_train = y_resampled

# Build LSTM model (more units is better, 0.2 dropout better than 0.1 or 0.3)
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=binary_focal_loss(), metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, shuffle=True)

# Predict on test set
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.1).astype(int).flatten()

# Accuracy
accuracy = np.mean(predicted_classes == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report 
print("Classification Report:")
print(classification_report(y_test, predicted_classes))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, predicted_classes)
print(cm)

# Actual vs Predicted Plot
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual Presence', marker='o', alpha=0.5)
plt.plot(dates_test, predicted_classes, label='Predicted Presence', marker='x', alpha=0.5)
plt.title('Actual vs Predicted Bluebottle Presence')
plt.xlabel('Date')
plt.ylabel('Presence (1 = Present, 0 = Absent)')
plt.legend()
plt.tight_layout()
plt.show()


