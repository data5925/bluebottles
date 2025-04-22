import os
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
from dask.distributed import Client
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.synthesizers import TimeSeriesSynthesizer
"""
run at ydata environment, look at https://docs.sdk.ydata.ai/latest/ more for information
"""


def main():
    client = Client(processes=False, n_workers=1, threads_per_worker=4)
    # ——— 1. License & data load —————————————————————————————
    os.environ['YDATA_LICENSE_KEY'] = '<your-token>'   # https://ydata.ai/ 
    df = pd.read_csv(r'D:\unsw\project5929\bluebottles\data\final_data.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # ——— 2. Create sliding windows ———————————————————————————
    window_size = 14
    feature_cols = [c for c in df.columns if c not in ['presence','bluebottles', 'beach.x']]

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[feature_cols].iloc[i:i+window_size].values)
        y.append(df['presence'].iloc[i+window_size])
    X = np.array(X)        # (n_samples, window_size, n_features)
    y = np.array(y)

    # ——— 3. One‑class SVM to pick “reliable” negatives —————————————
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    ocsvm.fit(X_pos.reshape(len(X_pos), -1))
    pred_neg = ocsvm.predict(X_neg.reshape(len(X_neg), -1))
    reliable_neg = X_neg[pred_neg == -1]  # outliers → assumed true negatives

    # ——— 4. Train TimeGAN on reliable negatives ———————————————————
    # Build a DataFrame with entity & time columns
    dfs = []
    for idx, win in enumerate(reliable_neg):
        tmp = pd.DataFrame(win, columns=feature_cols)
        tmp['time']   = np.arange(window_size)
        tmp['entity'] = idx
        dfs.append(tmp)
    df_windows = pd.concat(dfs, ignore_index=True)

    df_windows['time']   = df_windows['time'].astype(int)
    df_windows['entity'] = df_windows['entity'].astype(int)

    ds_neg = Dataset(df_windows)
    metadata_neg = Metadata(
        ds_neg,
        dataset_type='timeseries',
        dataset_attrs={'entities': ['entity'],
                       'sortbykey': 'time'}
    )

    # Override the dtype for `time` so it’s seen as numeric rather than categorical:
    metadata_neg.update_datatypes({'time': 'numerical'})

    print(ds_neg)
    print(metadata_neg)

    synth = TimeSeriesSynthesizer()
    synth.fit(ds_neg, metadata_neg)

    syn_ds = synth.sample(n_entities=len(reliable_neg)*2)
    syn_df = syn_ds.to_pandas()

    # Reconstruct windows from the synthetic dataset
    syn_wins = []
    for ent in syn_df['entity'].unique():
        part = syn_df[syn_df['entity'] == ent].sort_values('time')
        vals = part[feature_cols].values
        if vals.shape[0] == window_size:
            syn_wins.append(vals)
    X_syn_neg = np.array(syn_wins)

    # ——— 5. Combine & shuffle for LSTM training ———————————————————
    X_train = np.concatenate([X_pos, X_syn_neg], axis=0)
    y_train = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_syn_neg))])
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    print('finished')
    print(type(X_train), type(y_train))
    
    with open('trainData.npy', 'wb') as td:
        np.save(td, X_train)
        np.save(td, y_train)

    client.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

