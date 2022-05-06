from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError


class UnsupportedFileError(Exception):
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __str__(self) -> str:
        return f"unsupported file {self.filename}"


def load_file(
    filename: str, **kwargs: Any
) -> np.ndarray | torch.Tensor | pd.DataFrame | Image.Image:
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist")

    if filename.endswith(".npz"):
        result = np.load(filename)["arr_0"]
    elif filename.endswith(".npy"):
        result = np.load(filename)
    elif filename.endswith(".pt"):
        result = torch.load(filename)
    elif filename.endswith(".csv"):
        if "sep" in kwargs:
            kwargs.pop("sep")
        result = pd.read_csv(filename, sep=",", **kwargs)
    elif filename.endswith(".tsv") or filename.endswith(".txt"):
        if "sep" in kwargs:
            kwargs.pop("sep")
        result = pd.read_csv(filename, sep=r"\s+", **kwargs)
    elif filename.endswith(".xls") or filename.endswith(".xlsx"):
        result = pd.read_excel(filename, **kwargs)
    else:
        try:
            result = Image.open(filename, **kwargs)
        except UnidentifiedImageError:
            raise UnsupportedFileError(filename)
    return result


def load_dataframe(filename: str) -> pd.DataFrame:
    res = load_file(filename)
    if isinstance(res, pd.DataFrame):
        return res
    else:
        raise ValueError(f"{filename} is not a dataframe")