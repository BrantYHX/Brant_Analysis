import os
import pickle

def load_pickle(dataset, base_path=None):
    if base_path is None:
        base_path = os.path.expanduser("~/Documents/data_storage")

    path = os.path.join(base_path, f"{dataset}.pkl")

    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, dataset, base_path=None):
    if base_path is None:
        base_path = os.path.expanduser("~/Documents/data_storage")

    path = os.path.join(base_path, f"{dataset}.pkl")

    with open(path, "wb") as f:
        pickle.dump(data, f)
