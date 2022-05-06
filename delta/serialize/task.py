import os
from os import PathLike
from typing import IO, Union

import cloudpickle as pickle

from ..core.task import Task


def dump_task(file: Union[str, PathLike, IO[bytes]], task: Task):
    if isinstance(file, str):
        with open(file, mode="wb") as f:
            pickle.dump(task, f)
    elif isinstance(file, PathLike):
        filename = os.fspath(file)
        with open(filename, mode="wb") as f:
            pickle.dump(task, f)
    else:
        return pickle.dump(task, file)


def load_task(file: Union[str, PathLike, IO[bytes]]) -> Task:
    if isinstance(file, PathLike):
        filename = os.fspath(file)
        with open(filename, mode="rb") as f:
            res = pickle.load(f)
    elif isinstance(file, str):
        with open(file, mode="rb") as f:
            res = pickle.load(f)
    else:
        res = pickle.load(file)
    assert isinstance(res, Task), "File is not a task file"
    return res
