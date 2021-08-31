import os
from os import PathLike
from typing import IO, Union
from abc import ABC, abstractmethod

import cloudpickle as pickle

from delta.node import Node


class Task(ABC):
    def __init__(self, name: str, dataset: str):
        self.name = name
        self.dataset = dataset

    @classmethod
    def load(cls, file: Union[str, PathLike, IO[bytes]]) -> "Task":
        if isinstance(file, PathLike):
            filename = os.fspath(file)
            with open(filename, mode="rb") as f:
                return pickle.load(f)
        elif isinstance(file, str):
            with open(file, mode="rb") as f:
                return pickle.load(f)
        else:
            return pickle.load(file)

    def dump(self, file: Union[str, PathLike, IO[bytes]]):
        if isinstance(file, str):
            with open(file, mode="wb") as f:
                pickle.dump(self, f)
        elif isinstance(file, PathLike):
            filename = os.fspath(file)
            with open(filename, mode="wb") as f:
                pickle.dump(self, f)
        else:
            return pickle.dump(self, file)

    @property
    @abstractmethod
    def type(self) -> str:
        ...

    @abstractmethod
    def run(self, node: Node):
        ...
