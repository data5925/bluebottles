import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
df = pd.read_csv('final_data.csv')
df = df.drop(columns=['bluebottles'])

# Convert 'time' column to datetime objects
df['time'] = pd.to_datetime(df['time'])

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
column_names = df.columns.tolist()
features = column_names.copy()  # Create a copy of column_names
features.remove('presence')  # Remove 'presence' from the copy
features.remove('time')  # Remove 'time' from the features list
scaled_features = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled_features, columns=features)

# Data preparation for LSTM
def create_sequences(data, seq_length, presence_data): # Add presence_data as argument
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)].values # Get values of the DataFrame slice
        # Get y from the original presence data using the correct index
        y = presence_data.iloc[i+seq_length] # Access 'presence' from presence_data
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60 
# Pass df['presence'] to create_sequences function
x, y = create_sequences(scaled_df, seq_length, df['presence']) # Use .values to pass numpy array

# Addressing Class Imbalance
ros = RandomOverSampler(random_state=42)
# Reshape x to 2D before oversampling, but keep the 'time' feature
x_resampled, y_resampled = ros.fit_resample(x.reshape(x.shape[0], -1), df['presence'][seq_length+1:])
num_features = len(features) 
x_resampled = x_resampled.reshape(-1, seq_length, num_features)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshape the data for LSTM input
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Class weights for imbalanced data
class_weights = {0: 1, 1: 1} # Adjust if necessary based on class distribution in resampled data


# Create and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, class_weight=class_weights)


# Model Evaluation
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred_classes))
print(confusion_matrix(y_true, y_pred_classes))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Plot the presence prediction for the first 100 test samples
plt.figure(figsize=(12, 6))
plt.plot(y_true[:100], label='Actual')
plt.plot(y_pred_classes[:100], label='Predicted')
plt.title('LSTM Binary Classification - Presence (First 100 Test Samples)')
plt.xlabel('Sample')
plt.ylabel('Presence')
plt.legend()
plt.show()
