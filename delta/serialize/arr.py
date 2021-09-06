import os
from os import PathLike
from typing import IO, Union

import numpy as np


def load_arr(file: Union[str, PathLike, IO[bytes]]) -> np.ndarray:
    arr = np.load(file)["arr_0"]
    return arr


def dump_arr(file: Union[str, PathLike, IO[bytes]], arr: np.ndarray):
    if isinstance(file, str):
        with open(file, mode="wb") as f:
            np.savez_compressed(f, arr)
    elif isinstance(file, PathLike):
        filename = os.fspath(file)
        with open(filename, mode="wb") as f:
            np.savez_compressed(f, arr)
    else:
        np.savez_compressed(file, arr)
