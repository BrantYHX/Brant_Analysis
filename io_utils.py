import os
import pickle


class _ChunkedWriter:
    """Wraps a file and breaks large write() calls into 1 MB chunks.
    Needed on Windows SMB shares where a single large WriteFile call fails
    with OSError [Errno 22] Invalid argument."""
    def __init__(self, file, chunk_size=1 << 20):
        self._file = file
        self._chunk = chunk_size

    def write(self, data):
        for i in range(0, len(data), self._chunk):
            self._file.write(data[i:i + self._chunk])
        return len(data)

    def flush(self):
        self._file.flush()


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
        pickle.dump(data, _ChunkedWriter(f))
