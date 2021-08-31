from pathlib import Path
from typing import IO, Union

from .horizontol import HorizontolTask
from .task import Task

__all__ = ["Task", "HorizontolTask", "load"]


def load(file: Union[str, Path, IO[bytes]]) -> Task:
    return Task.load(file)
