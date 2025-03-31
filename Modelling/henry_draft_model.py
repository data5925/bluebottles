import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import random
import os

# Set seed for consistent results
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
tf.random.set_seed(42)

# Load data
data = pd.read_csv('data/final_data.csv')
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)

# Separate features and target
features = data.drop(columns=['presence', 'bluebottles'])  
target = data['presence']  

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Set window size and create sequences
window_size = 30
X, y, target_dates = [], [], []
for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i])
    y.append(target[i])
    target_dates.append(data.index[i])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)
target_dates = np.array(target_dates)

# Time Series Cross-Validation 
tscv = TimeSeriesSplit(n_splits=5)
fold_accuracies = []
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"\nFold {fold + 1}/{tscv.n_splits}")
    
    # Split data for this fold
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Apply SMOTE on training fold
    X_train_flat = X_train_fold.reshape(X_train_fold.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train_fold)

    # Reshape back to 3D for LSTM
    X_train_fold = X_resampled.reshape((X_resampled.shape[0], window_size, X_train_fold.shape[2]))
    y_train_fold = y_resampled

    # Build the LSTM model for each fold
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train_fold.shape[1], X_train_fold.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model for this fold
    history = model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, validation_split=0.1, shuffle=True, verbose=0)

    # Predict and evaluate fold
    predictions = model.predict(X_test_fold)
    predicted_classes = (predictions >= 0.5).astype(int).flatten()

    # Calculate fold accuracy
    accuracy = np.mean(predicted_classes == y_test_fold)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}")

# Mean Accuracy Across Folds
mean_accuracy = np.mean(fold_accuracies)
print(f"\nMean Accuracy Across Folds: {mean_accuracy:.2f}")

# ------------------------------------
# Final training on full data after CV
# ------------------------------------
X_train_flat = X.reshape(X.shape[0], -1)
X_resampled, y_resampled = smote.fit_resample(X_train_flat, y)

# Reshape back for LSTM
X_train_final = X_resampled.reshape((X_resampled.shape[0], window_size, X.shape[2]))
y_train_final = y_resampled

# Build final model
final_model = Sequential()
final_model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train_final.shape[1], X_train_final.shape[2])))
final_model.add(Dropout(0.2))
final_model.add(LSTM(units=128))
final_model.add(Dropout(0.2))
final_model.add(Dense(1, activation='sigmoid'))

final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the final model
final_model.fit(X_train_final, y_train_final, epochs=20, batch_size=32, validation_split=0.1, shuffle=True)

# Predict on test Set
predictions = final_model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int).flatten()

# Evaluate Final Model
accuracy = np.mean(predicted_classes == y_test)
print(f'\nFinal Model Accuracy on Test Set: {accuracy:.2f}')

# Confusion matrix
cm = confusion_matrix(y_test, predicted_classes)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

# Plot actual vs predicted presence
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual Presence', alpha=0.5)
plt.plot(dates_test, predicted_classes, label='Predicted Presence', alpha=0.5)
plt.title('Actual vs Predicted Bluebottle Presence')
plt.xlabel('Date')
plt.ylabel('Presence (1 = Present, 0 = Absent)')
plt.legend()
plt.tight_layout()
plt.show()