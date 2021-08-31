from pathlib import Path
from typing import IO, Union

import numpy as np


def load_arr(file: Union[str, Path, IO[bytes]]) -> np.ndarray:
    arr = np.load(file)["arr_0"]
    return arr


def dump_arr(file: Union[str, Path, IO[bytes]], arr: np.ndarray):
    np.savez_compressed(file, arr)
