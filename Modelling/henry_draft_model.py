import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

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

# Create sequences
window_size = 30
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

# Apply SMOTE to balance the classes 
X_train_flat = X_train.reshape((X_train.shape[0], -1))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)

# Reshape back to 3D for LSTM 
X_train = X_resampled.reshape((X_resampled.shape[0], window_size, X.shape[2]))
y_train = y_resampled

# Build LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, shuffle=True)

# Predict on test set
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int).flatten()

# Accuracy
accuracy = np.mean(predicted_classes == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report 
print("Classification Report:")
print(classification_report(y_test, predicted_classes))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

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


