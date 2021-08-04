import logging
import os.path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Iterable, Optional, IO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .node import Node


class DebugDataset(Dataset):
    def __init__(self, dataset_path: str, preprocess: Callable = None):
        self._data: torch.Tensor
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
    def __init__(self, task_id: int):
        self._logger = logging.getLogger(__name__)

        self._temp_dir = TemporaryDirectory()

        self._task_id = task_id
        self._state_count = 0
        self._weight_count = 0

    def state_file(self) -> str:
        return os.path.join(self._temp_dir.name, f"{self._task_id}.state.{self._state_count}")

    def weight_file(self) -> str:
        return os.path.join(
            self._temp_dir.name, f"{self._task_id}.weight.{self._weight_count}"
        )

    def new_dataloader(
        self, dataset: str, dataloader: Dict[str, Any], preprocess: Callable
    ) -> Iterable:
        data = DebugDataset(dataset, preprocess)
        return DataLoader(data, **dataloader)

    def download_state(self, dst: IO[bytes]) -> bool:
        filename = self.state_file()
        if os.path.exists(filename):
            self._logger.info(f"load state {filename} for task {self._task_id}")
            with open(filename, mode="rb") as f:
                shutil.copyfileobj(f, dst)
            return True
        else:
            self._logger.info(f"initial state for task {self._task_id}")
            return False

    def upload_state(self, file: IO[bytes]):
        self._state_count += 1
        filename = self.state_file()
        with open(filename, mode="wb") as f:
            shutil.copyfileobj(file, f)
        self._logger.info(f"dump state {filename} for task {self._task_id}")

    def upload_result(self, file: IO[bytes]):
        self._weight_count += 1
        filename = self.weight_file()
        with open(filename, mode="wb") as f:
            shutil.copyfileobj(file, f)
        self._logger.info(f"upload weight {filename} for task {self._task_id}")

    def download_weight(self, dst: IO[bytes]) -> bool:
        filename = self.weight_file()
        if os.path.exists(filename):
            self._logger.info(f"download weight {filename} for task {self._task_id}")
            with open(filename, mode="rb") as f:
                shutil.copyfileobj(f, dst)
            return True
        else:
            self._logger.info(f"initial weight for task {self._task_id}")
            return False
