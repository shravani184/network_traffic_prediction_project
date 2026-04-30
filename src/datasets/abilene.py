import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch

def _read_matrix(path, delimiter="auto"):
    if delimiter == "auto":
        for sep in [",", "\t", " "]:
            try:
                df = pd.read_csv(path, sep=sep, header=None)
                if df.shape[1] > 1:
                    return df.values.astype(float)
            except:
                continue
        return pd.read_csv(path, sep="\s+", header=None).values
    else:
        return pd.read_csv(path, sep=delimiter, header=None).values

def make_sliding_windows(data, window, horizon, use_multivariate, column_id):
    if not use_multivariate:
        data = data[:, [column_id]]
    X, y = [], []
    for i in range(window, len(data) - horizon + 1):
        X.append(data[i - window:i])
        y.append(data[i + horizon - 1, 0])
    return np.array(X), np.array(y).reshape(-1, 1)

class AbileneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_dataset(dataset_path, delimiter, window_size, horizon, use_multivariate, column_id, normalize, train_ratio, val_ratio):
    data = _read_matrix(dataset_path, delimiter)
    X, y = make_sliding_windows(data, window_size, horizon, use_multivariate, column_id)
    total = len(X)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    scalers = {}
    if normalize:
        sc_x = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
        sc_y = StandardScaler().fit(y_train)
        X_train = sc_x.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = sc_x.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = sc_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        y_train = sc_y.transform(y_train)
        y_val = sc_y.transform(y_val)
        y_test = sc_y.transform(y_test)
        scalers = {"x": sc_x, "y": sc_y}

    return {
        "train": AbileneDataset(X_train, y_train),
        "val": AbileneDataset(X_val, y_val),
        "test": AbileneDataset(X_test, y_test),
        "scalers": scalers
    }
