from __future__ import annotations

import errno
import os
import random
from typing import Any, List, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from ..core.task import DataFormat, DataLocation, InputGraphNode
from .file import load_file


class Dataset(InputGraphNode):
    def __init__(
        self,
        dataset: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="",
            location=DataLocation.CLIENT,
            src=None,
            format=DataFormat.DATASET,
            filename=dataset,
            default=None,
            **kwargs,
        )


class FileDataset(TorchDataset):
    def __init__(self, filename: str) -> None:
        result = load_file(filename)
        if isinstance(result, Image.Image):
            raise ValueError("file dataset does not support image file")
        self._result = result

    def __getitem__(self, index):
        if isinstance(self._result, pd.DataFrame):
            item = self._result.iloc[index]
        else:
            item = self._result[index]

        return item

    def __len__(self) -> int:
        return len(self._result)


class DirectoryDataset(TorchDataset):
    def __init__(self, directory: str) -> None:
        self._xs = []
        self._ys = []
        root, dirnames, filenames = next(os.walk(directory))
        if len(filenames) > 0 and len(dirnames) == 0:
            self._xs.extend([os.path.join(root, filename) for filename in filenames])
        elif len(filenames) == 0 and len(dirnames) > 0:
            for dirname in dirnames:
                sub_root, sub_dirs, sub_files = next(
                    os.walk(os.path.join(root, dirname))
                )
                if len(sub_files) > 0 and len(sub_dirs) == 0:
                    self._xs.extend(
                        [os.path.join(sub_root, filename) for filename in sub_files]
                    )
                    self._ys.extend([dirname] * len(sub_files))
                else:
                    raise ValueError(
                        "directory can only contain files or sub directory that contains file"
                    )
        else:
            raise ValueError(
                "directory can only contain files or sub directory that contains file"
            )

    def __getitem__(self, index):
        filename = self._xs[index]
        x = load_file(filename)
        y = None
        if len(self._ys) > 0:
            y = self._ys[index]
        if y is not None:
            return x, y
        return x

    def __len__(self) -> int:
        return len(self._xs)


class IndexLookUpDataset(TorchDataset):
    def __init__(self, indices: List[int], dataset: TorchDataset) -> None:
        self._indices = indices
        self._dataset = dataset

    def __getitem__(self, index):
        i = self._indices[index]
        return self._dataset[i]

    def __len__(self) -> int:
        return len(self._indices)


def split_dataset(
    dataset: TorchDataset, frac: float, seed: int = 0
) -> Tuple[IndexLookUpDataset, IndexLookUpDataset]:
    assert frac > 0 and frac < 1

    size = len(dataset)  # type: ignore
    frac_count = int(size * frac)
    indices = list(range(size))

    state = random.Random(seed)
    state.shuffle(indices)

    left_indices, right_indices = indices[:frac_count], indices[frac_count:]
    return IndexLookUpDataset(left_indices, dataset), IndexLookUpDataset(
        right_indices, dataset
    )


def load_dataset(
    dataset_name: str,
) -> TorchDataset | Tuple[TorchDataset, TorchDataset]:
    if not os.path.exists(dataset_name):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_name)
    if os.path.isfile(dataset_name):
        dataset = FileDataset(dataset_name)
        return dataset
    else:
        train_path = os.path.join(dataset_name, "train")
        val_path = os.path.join(dataset_name, "val")
        if (
            os.path.exists(train_path)
            and os.path.isdir(train_path)
            and os.path.exists(val_path)
            and os.path.isdir(val_path)
        ):
            train_root, _, train_files = next(os.walk(train_path))
            val_root, _, val_files = next(os.walk(val_path))
            if len(train_files) == 1 and len(val_files) == 1:
                train_dataset = FileDataset(os.path.join(train_root, train_files[0]))
                val_dataset = FileDataset(os.path.join(val_root, val_files[0]))
            else:
                train_dataset = DirectoryDataset(train_path)
                val_dataset = DirectoryDataset(val_path)
            return train_dataset, val_dataset
        else:
            dataset = DirectoryDataset(dataset_name)
            return dataset
