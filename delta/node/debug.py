import logging
import os.path
from tempfile import TemporaryDirectory
from typing import Dict, Any, Callable, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .node import Node


class DebugDataset(Dataset):
    def __init__(self, dataset_path: str, preprocess: Callable = None):
        self._data = None
        self._preprocess = preprocess
        if not os.path.exists(dataset_path):
            raise RuntimeError(f"dataset path {dataset_path} does not exist")
        if os.path.isfile(dataset_path):
            if dataset_path.endswith(".npz"):
                data = np.load(dataset_path)["arr_0"]
                self._data = torch.from_numpy(data).float()
            elif dataset_path.endswith(".npy"):
                data = np.load(dataset_path)
                self._data = torch.from_numpy(data).float()
            elif dataset_path.endswith(".pt"):
                self._data = torch.load(dataset_path)
            else:
                raise RuntimeError("unsupported dataset file format")
        else:
            raise RuntimeError("dataset path can only be a file now")

    def __getitem__(self, index):
        item = self._data[index]
        if self._preprocess is not None:
            item = self._preprocess(item)
        return item

    def __len__(self):
        return len(self._data)

class DebugNode(Node):
    def __init__(self):
        self._logger = logging.getLogger(__name__)

        self._temp_dir = TemporaryDirectory()

        self._state_count = 0
        self._weight_count = 0

    def state_file(self, task_id: int) -> str:
        return os.path.join(self._temp_dir.name, f"{task_id}.state.{self._state_count}")

    def weight_file(self, task_id: int) -> str:
        return os.path.join(self._temp_dir.name, f"{task_id}.weight.{self._weight_count}")

    def new_dataloader(self, dataset: str, dataloader: Dict[str, Any], preprocess: Callable) -> Iterable:
        dataset = DebugDataset(dataset, preprocess)
        return DataLoader(dataset, **dataloader)

    def download_state(self, task_id: int) -> Optional[bytes]:
        filename = self.state_file(task_id)
        if os.path.exists(filename):
            self._logger.info(f"load state {filename} for task {task_id}")
            with open(filename, mode="rb") as f:
                return f.read()
        else:
            self._logger.info(f"initial state for task {task_id}")
            return None

    def upload_state(self, task_id: int, data: bytes):
        self._state_count += 1
        filename = self.state_file(task_id)
        with open(filename, mode="wb") as f:
            f.write(data)
        self._logger.info(f"dump state {filename} for task {task_id}")

    def upload_weight(self, task_id: int, data: bytes):
        self._weight_count += 1
        filename = self.weight_file(task_id)
        with open(filename, mode="wb") as f:
            f.write(data)
        self._logger.info(f"upload weight {filename} for task {task_id}")

    def download_weight(self, task_id) -> Optional[bytes]:
        filename = self.weight_file(task_id)
        if os.path.exists(filename):
            self._logger.info(f"download weight {filename} for task {task_id}")
            with open(filename, mode="rb") as f:
                return f.read()
        else:
            self._logger.info(f"initial weight for task {task_id}")
            return None
