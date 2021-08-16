import os
from typing import Any, Callable, Union, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader


class UnsupportedFileError(Exception):
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __str__(self) -> str:
        return f"unsupported file {self.filename}"


def load_file(
    filename: str,
) -> Union[np.ndarray, torch.Tensor, pd.DataFrame, Image.Image]:
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist")

    if filename.endswith(".npz"):
        result = np.load(filename)["arr_0"]
    elif filename.endswith(".npy"):
        result = np.load(filename)
    elif filename.endswith(".pt"):
        result = torch.load(filename)
    elif filename.endswith(".csv"):
        result = pd.read_csv(filename, sep=",")
    elif filename.endswith(".tsv") or filename.endswith(".txt"):
        result = pd.read_csv(filename, sep=r"\s+")
    elif filename.endswith(".xls") or filename.endswith(".xlsx"):
        result = pd.read_excel(filename)
    else:
        try:
            result = Image.open(filename)
        except UnidentifiedImageError:
            raise UnsupportedFileError(filename)
    return result


class FileDataset(Dataset):
    def __init__(self, filename: str, preprocess: Callable = None) -> None:
        result = load_file(filename)
        if isinstance(result, Image.Image):
            raise ValueError("file dataset does not support image file")
        self._result = result
        self._preprocess = preprocess

    def __getitem__(self, index):
        if isinstance(self._result, pd.DataFrame):
            item = self._result.iloc[index]
        else:
            item = self._result[index]

        if self._preprocess is not None:
            item = self._preprocess(item)
        return item

    def __len__(self) -> int:
        return len(self._result)


class DirectoryDataset(Dataset):
    def __init__(self, directory: str, preprocess: Callable = None) -> None:
        self._xs = []
        self._ys = []
        self._preprocess = preprocess
        root, dirnames, filenames = next(os.walk(directory))
        if len(filenames) > 0 and len(dirnames) == 0:
            self._xs.extend([os.path.join(root, filename) for filename in filenames])
        elif len(filenames) == 0 and len(dirnames) > 0:
            for dirname in dirnames:
                sub_root, sub_dirs, sub_files = next(os.walk(os.path.join(root, dirname)))
                if len(sub_files) > 0 and len(sub_dirs) == 0:
                    self._xs.extend(
                        [os.path.join(sub_root, filename) for filename in sub_files]
                    )
                    self._ys.extend([dirname] * len(sub_files))
                else:
                    raise ValueError("directory can only contain files or sub directory that contains file")
        else:
            raise ValueError("directory can only contain files or sub directory that contains file")
    
    def __getitem__(self, index):
        filename = self._xs[index]
        x = load_file(filename)
        y = None
        if len(self._ys) > 0:
            y = self._ys[index]
        if self._preprocess is not None:
            if y is not None:
                x, y = self._preprocess(x, y)
            else:
                x = self._preprocess(x)
        if y is not None:
            return x, y
        return x

    def __len__(self) -> int:
        return len(self._xs)


def new_dataloader(dataset_name: str, dataloader_cfg: Dict[str, Any], preprocess: Callable):
    if os.path.isfile(dataset_name):
        dataset = FileDataset(dataset_name, preprocess)
    elif os.path.isdir(dataset_name):
        dataset = DirectoryDataset(dataset_name, preprocess)
    else:
        raise ValueError("dataset should be a file or a directory")
    dataloader = DataLoader(dataset, **dataloader_cfg)
    return dataloader
