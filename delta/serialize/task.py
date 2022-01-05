from os import PathLike
from typing import IO, Union

from delta.task import Task


def dump_task(file: Union[str, PathLike, IO[bytes]], task: Task):
    task.dump(file)


def load_task(file: Union[str, PathLike, IO[bytes]]) -> Task:
    return Task.load(file)
