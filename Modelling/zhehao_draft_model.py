import os
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from ydata.synthesizers import TimeSeriesSynthesizer
from ydata.metadata import Metadata
from ydata.dataset import Dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 0. Configuration ───────────────────────────────────────────────────────────
# set your YData SDK token: (you can register one at https://dashboard.ydata.ai/)
os.environ["YDATA_LICENSE_KEY"] = 'e357cac2-da58-414f-b5b4-62ede1723689'  

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ─── 1. LOAD & PREPROCESS ───────────────────────────────────────────────────────
# 1.1 Load
df = pd.read_csv('bluebottles/data/final_data.csv', parse_dates=["time"])
df.set_index("time", inplace=True)

# 1.2 Features & target
feature_cols = [c for c in df.columns if c not in ["presence", "bluebottles"]]
X_raw = df[feature_cols].values
y_raw = df["presence"].values

# 1.3 Normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# 1.4 Create sliding windows
window_size = 30
X_windows, y_windows = [], []
for i in range(window_size, len(X_scaled)):
    X_windows.append(X_scaled[i-window_size : i])
    y_windows.append(y_raw[i])
X_windows = np.array(X_windows)      # shape = (n_samples, window_size, n_features)
y_windows = np.array(y_windows)

# ─── 2. STEP 1: ONE‑CLASS SVM ON POSITIVES ───────────────────────────────────────
# Train on only the positive windows
pos_windows = X_windows[y_windows == 1]
pos_flat = pos_windows.reshape((pos_windows.shape[0], -1))
ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
ocsvm.fit(pos_flat)

# Get OC‑SVM scores on all data (for later analysis if desired)
all_flat = X_windows.reshape((X_windows.shape[0], -1))
ocsvm_scores = ocsvm.decision_function(all_flat)

# ─── 3. STEP 2a: ARTIFICIAL “BACKGROUND” NEGATIVES ─────────────────────────────
# Following Vert & Vert (2006), sample negatives from a Gaussian background
# fitted to the positive class density level sets :contentReference[oaicite:0]{index=0}
mean_pos = np.mean(pos_flat, axis=0)
cov_pos  = np.cov(pos_flat, rowvar=False)
n_artificial = len(pos_flat)
art_neg_flat = np.random.multivariate_normal(mean_pos, cov_pos, size=n_artificial)
art_neg = art_neg_flat.reshape((n_artificial, window_size, X_windows.shape[2]))

# ─── 4. STEP 2b: TimeGAN SYNTHESIS (YData‑SDK) ──────────────────────────
# Prepare a long‐format DataFrame so YData SDK can train a time-series GAN
# We treat each window as an “entity” with its own time index
entities = np.repeat(np.arange(len(pos_flat)), window_size)
times    = np.tile(np.arange(window_size), len(pos_flat))
flat_vals = pos_flat  # we train the GAN on *positive* windows so that sampling gives similar structure
flat_vals = flat_vals.reshape((-1, X_windows.shape[2]))

gan_df = pd.DataFrame(flat_vals, columns=feature_cols)
gan_df["entity"] = entities
gan_df["time"]   = times

gan_dataset = Dataset(gan_df)
# Build metadata for a single-entity, multivariate time-series
dataset_attrs={
        "sortbykey": "time",
        "entities": ["entity"]
    }
    
metadata = Metadata(gan_dataset, dataset_type='timeseries', dataset_attrs=dataset_attrs)

# Train and sample
synth = TimeSeriesSynthesizer()
synth.fit(gan_dataset, metadata)  
n_gan = n_artificial
synth_samples = synth.sample(n_entities=n_gan)  
# Convert back to windows
synth_vals = synth_samples[feature_cols].values
ctgan_neg = synth_vals.reshape((n_gan, window_size, X_windows.shape[2]))

# ─── 5. STEP 3: COMBINE & EVALUATE ──────────────────────────────────────────────
# Positives + Artificial Negatives + GAN Negatives
X_combo = np.vstack([pos_windows, art_neg, ctgan_neg])
y_combo = np.hstack([np.ones(len(pos_windows)), np.zeros(len(art_neg) + len(ctgan_neg))])

# Split for model training
X_tr, X_te, y_tr, y_te = train_test_split(
    X_combo, y_combo, test_size=0.2, random_state=42, shuffle=True
)

# ─── 6. TRAIN A SIMPLE LSTM CLASSIFIER ─────────────────────────────────────────
model = Sequential([
    LSTM(64, input_shape=(window_size, X_windows.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_tr, y_tr,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

# ─── 7. FINAL EVALUATION & PLOTS ────────────────────────────────────────────────
# Predict
y_pred = (model.predict(X_te) >= 0.5).astype(int).flatten()

# Metrics
print(classification_report(y_te, y_pred))
cm = confusion_matrix(y_te, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"])
plt.title("Confusion Matrix")
plt.show()


"""
TypeError, still investigating what is going on.
---> 90 synth.fit(gan_dataset, metadata)
TypeError: Type <class 'dask_expr._collection.DataFrame'> is not supported by 'to_dask' helper method.
"""





