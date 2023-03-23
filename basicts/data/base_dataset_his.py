from torch.utils.data import Dataset
import torch
from utils.serialization import load_pkl
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class BaseDataset(Dataset):
    def __init__(self, raw_file_path: str, index_file_path: str, mode: str, **kwargs) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'test'], "error mode"
        # read raw data (normalized)
        data = load_pkl(raw_file_path)
        raw_data = data['raw_data']

        self.data = torch.from_numpy(raw_data).float()               # L, N, C
        # read index
        self.index = load_pkl(index_file_path)[mode]
        print("index is ok")

    def __getitem__(self, index: int) -> tuple:
        """get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """
        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            future_data = self.data[idx[1]:idx[2]]
            history_data0 = torch.zeros_like(future_data)
            history_data1 = torch.zeros_like(future_data)
            history_data2 = self.data[idx[0]:idx[1]]
            if idx[1]-7*288 >= 0:
                history_data0 = self.data[idx[1]-7*288:idx[2]-7*288]
            if idx[1]-288 >= 0:
                history_data1 = self.data[idx[1]-288:idx[2]-288]
        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]    # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]

        return future_data, history_data0, history_data1, history_data2

    def __len__(self):
        """dataset length

        Returns:
            int: dataset length
        """
        return len(self.index)
