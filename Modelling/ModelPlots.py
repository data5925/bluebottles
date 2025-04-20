import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, os, math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
import tensorflow.keras.backend as K


def featureImportance(features, X_test, y_test, threshold, model):
  # plot feature permutation importance
    feature_names = list(features.columns)
    n_features    = X_test.shape[2]
    threshold     = 0.1

    # 1) Baseline accuracy on original X_test
    probs_base   = model.predict(X_test)
    preds_base   = (probs_base >= threshold).astype(int).flatten()
    acc_base     = np.mean(preds_base == y_test)
    print(f"Baseline test accuracy: {acc_base:.3f}")

    # 2) Permutation importances
    importances = []

    for i, fname in enumerate(feature_names):
        # Copy X_test
        X_perm = X_test.copy()
        # Permute values of feature i across the samples (keeping window intact)
        perm_idx              = np.random.permutation(X_perm.shape[0])
        X_perm[:, :, i]       = X_perm[perm_idx, :, i]
        # Evaluate permuted accuracy
        probs_perm            = model.predict(X_perm)
        preds_perm            = (probs_perm >= threshold).astype(int).flatten()
        acc_perm              = np.mean(preds_perm == y_test)
        # Importance = drop in accuracy
        importances.append(acc_base - acc_perm)
        print(f"Feature: {fname:15s}  Δacc = {importances[-1]:.4f}")

    # 3) Plot importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Decrease in accuracy")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.show()


def featureDistribution(features, target):
    # Plot the distribution of input features for both presence and absence cases of training data
    # 1) Split raw features into train/test
    feat_train, feat_test, tar_train, tar_test = train_test_split(
        features, target,
        test_size=0.2,
        shuffle=False
)

    # 2) Scale training features
    scaler = MinMaxScaler()
    feat_train_scaled = pd.DataFrame(
        scaler.fit_transform(feat_train),
        index=feat_train.index,
        columns=feat_train.columns
    )
    feat_train_scaled['presence'] = tar_train.values

    # 3) Prepare subplot grid
    feature_cols = feat_train_scaled.columns.drop('presence')
    n_feats = len(feature_cols)
    n_cols = 4
    n_rows = math.ceil(n_feats / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    # 4) Loop features and plot
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.kdeplot(
            data=feat_train_scaled,
            x=col,
            hue='presence',
            multiple='layer',
            common_norm=False,
            fill=True,
            alpha=0.5,
            ax=ax
        )
        ax.set_title(f'{col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend(title='Presence', labels=['Absent (0)', 'Present (1)'])

    # 5) Hide any unused subplots
    for j in range(n_feats, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()