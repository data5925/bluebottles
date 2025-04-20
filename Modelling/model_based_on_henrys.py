import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, os, math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import tensorflow.keras.backend as K
from ModelPlots import featureImportance, featureDistribution


# focal loss function (alpha = 0.75 produces best balance)
def binary_focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32') 
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
data = pd.read_csv(r'D:\unsw\project5929\bluebottles\data\final_data.csv')
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)
# data = data[data['beach.x'] == 0] # when uncommenting this line, remove beach.x in features and change threshold back to 0.5

# drop features based on the distribution of input features
feature_cols = [c for c in data.columns if c not in ["presence", "bluebottles", "crt_temp","wave_fp","wave_cos_dir","wave_dir"]]
features = data[feature_cols]
target = data['presence']

# Plot the distribution of input features for both presence and absence cases of training data
featureDistribution(features, target)  # or call function: Plot the distribution of input features for both presence and absence cases of training data

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

# define class-weights
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
cw = dict(enumerate(weights))

# Build BiLSTM model 
model = Sequential([
  Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
  Dropout(0.2),
  Bidirectional(LSTM(128)),
  Dropout(0.2),
  Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=binary_focal_loss(alpha=0.75, gamma=2.0), metrics=['accuracy'])

# Train model
early_stop = EarlyStopping(
    monitor='val_loss',      
    patience=5,              
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,               
    batch_size=32,
    validation_split=0.1,
    shuffle=True,
    callbacks=[early_stop],
    class_weight=cw
)

# Predict on test set
predictions = model.predict(X_test)
threshold = 0.1
predicted_classes = (predictions >= threshold).astype(int).flatten()

# Classification Report 
print("Classification Report:")
print(classification_report(y_test, predicted_classes))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, predicted_classes)
labels = ['Absence (0)', 'Presence (1)']
plt.figure(figsize=(6, 5))
sns.heatmap(cm,
            annot=True,         
            fmt='d',            
            cmap='Blues',       
            xticklabels=labels,
            yticklabels=labels,
            cbar=False)         
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
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

# plot feature permutation importance
featureImportance(features, X_test, y_test, threshold, model)  # or call function: plot feature permutation importance


"""
Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.66      0.78       635
           1       0.06      0.47      0.11        30

    accuracy                           0.65       665
   macro avg       0.51      0.56      0.44       665
weighted avg       0.92      0.65      0.75       665

confusion matrix: see confusionMatrix.png
distribution of presence vs absence plots: see distribution presence vs absence.png
feature permutation importance plots: see featureImportance.png

Ydata Module Issues: see zhehao_draft_model.py comments
Ydata package imcompatibility: errors occur when tensorflow and ydata-sdk coexist in the conda environment
"""
