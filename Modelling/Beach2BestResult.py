import pandas as pd
import numpy as np
import random, os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# ========== Loss Functions ========== #
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    return focal_loss_fixed

def combo_loss(gamma=2.0, alpha=0.25, focal_weight=0.6):
    fl = focal_loss(gamma=gamma, alpha=alpha)
    bce = tf.keras.losses.BinaryCrossentropy()
    def loss_fn(y_true, y_pred):
        return focal_weight * fl(y_true, y_pred) + (1 - focal_weight) * bce(y_true, y_pred)
    return loss_fn

# ========== Utility Functions ========== #
def evaluate_thresholds(y_true, y_prob, beach_id):

    thresholds = np.arange(0.01, 0.5, 0.01)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        results.append((threshold, precision, recall, f1))
    df_results = pd.DataFrame(results, columns=["Threshold", "Precision", "Recall", "F1 Score"])
    best_idx = df_results["F1 Score"].idxmax()
    best_threshold = df_results.loc[best_idx, "Threshold"]
    best_f1 = df_results.loc[best_idx, "F1 Score"]
    y_pred_best = (y_prob > best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    report = classification_report(y_true, y_pred_best)
    print(f"✅ Beach {beach_id} | Best Threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    return df_results, best_threshold, y_pred_best

def create_sequences(X, y, time_steps=14):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def balance_classes(X, y):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_samples = min(len(pos_idx), len(neg_idx))
    pos_sampled = np.random.choice(pos_idx, n_samples, replace=False)
    neg_sampled = np.random.choice(neg_idx, n_samples, replace=False)
    idx = np.concatenate([pos_sampled, neg_sampled])
    idx = np.sort(idx)  # 保持时间顺序
    return X[idx], y[idx]

# ========== Data Preparation ========== #
df = pd.read_csv("data/final_data.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.rename(columns={'beach.x': 'beach'})
df = df.sort_values(by='time').reset_index(drop=True)
df['dayofyear'] = df['time'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365)



# 特征组合
new_features = [
    'crt_u', 'crt_v',
    'wave_hs', 'wnd_sfcWindspeed',
     'wnd_uas', 'wnd_vas', 'sin_day', 'cos_day'
]
target = 'presence'
time_steps = 14

# ========== 主流程 ========== #
beach_ids = [2]
for beach_id in beach_ids:
    print(f"\n🏖 Processing beach {beach_id}...\n" + "-"*40)
    df_beach = df[df['beach'] == beach_id].copy().reset_index(drop=True)

    # 数据预处理
    X = df_beach[new_features].values
    y = df_beach[target].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    test_time = df_beach['time'][time_steps:].reset_index(drop=True)

    # 时间序列分割
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    test_time = test_time[split_idx:]

    # 类别平衡处理
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    # 海滩特定配置
    lstm_units = [128, 128]
    dropout_rate = 0.2
    loss_fn = combo_loss(focal_weight=0.65)

    # ========== 交叉验证 ========== #
    tscv = TimeSeriesSplit(n_splits=3)
    cv_metrics = []
    print("\n Starting Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_bal)):
        print(f"\n Fold {fold+1}/{tscv.n_splits}")
        X_train_fold, X_val_fold = X_train_bal[train_idx], X_train_bal[val_idx]
        y_train_fold, y_val_fold = y_train_bal[train_idx], y_train_bal[val_idx]

        # 模型构建
        model = Sequential([
            Bidirectional(LSTM(lstm_units[0], return_sequences=True),
                          input_shape=(time_steps, len(new_features))),
            BatchNormalization(),
            Dropout(dropout_rate),
            LSTM(lstm_units[1]),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        
        # early stop设置
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        
        # 模型训练
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # 验证集评估
        y_val_prob = model.predict(X_val_fold).flatten()
        _, best_threshold, y_val_pred = evaluate_thresholds(y_val_fold, y_val_prob, beach_id)
        f1 = f1_score(y_val_fold, y_val_pred)
        cv_metrics.append(f1)
        print(f" Fold {fold+1} Validation F1: {f1:.4f}")

    # 输出交叉验证结果
    print(f"\n Beach {beach_id} CV Results:")
    print(f"Mean F1: {np.mean(cv_metrics):.4f} (±{np.std(cv_metrics):.4f})")

    # ========== 全量训练 ========== #
    print("\n Training Final Model...")
    final_model = Sequential([
        Bidirectional(LSTM(lstm_units[0], return_sequences=True),
                      input_shape=(time_steps, len(new_features))),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(lstm_units[1]),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    final_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    early_stop_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    final_model.fit(
        X_train_bal, y_train_bal,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop_final],
        verbose=0
    )

    # ========== 测试集评估 ========== #
    print("\n Final Evaluation on Test Set:")
    y_prob = final_model.predict(X_test).flatten()
    threshold_df, best_threshold, y_pred_label = evaluate_thresholds(y_test, y_prob, beach_id)

    # 可视化结果
    df_time = pd.DataFrame({
        "Time": test_time,
        "Actual": y_test,
        "Predicted": y_pred_label,
        "Predicted Prob.": y_prob
    })
    plt.figure(figsize=(16, 5))
    plt.plot(df_time["Time"], df_time["Actual"], label="Actual", marker='o', alpha=0.6)
    plt.plot(df_time["Time"], df_time["Predicted"], label="Predicted", marker='x', alpha=0.6)
    plt.plot(df_time["Time"], df_time["Predicted Prob."], label="Predicted Prob.", alpha=0.5)
    plt.axhline(best_threshold, color='red', linestyle=':', label=f'Threshold = {best_threshold:.2f}')
    plt.title(f"Beach {beach_id} - Actual vs Predicted Bluebottle Presence")
    plt.xlabel("Time")
    plt.ylabel("Label / Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
