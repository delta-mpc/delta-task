from pathlib import Path
from typing import IO, Union

from .horizontal import HorizontalTask
from .task import Task

__all__ = ["Task", "HorizontalTask", "load"]


def load(file: Union[str, Path, IO[bytes]]) -> Task:
    return Task.load(file)
