from abc import ABC, abstractmethod
from typing import List, IO, BinaryIO

import numpy as np

from ..node import Node


class Task(ABC):
    @classmethod
    @abstractmethod
    def loads_cfg(cls, data: bytes):
        ...

    @classmethod
    @abstractmethod
    def load_cfg(cls, file: IO[bytes]):
        ...

    @abstractmethod
    def dumps_cfg(self) -> bytes:
        ...

    @abstractmethod
    def dump_cfg(self, file: IO[bytes]) -> bytes:
        ...

    @abstractmethod
    def loads_weight(self, data: bytes):
        ...

    @abstractmethod
    def load_weight(self, file: IO[bytes]):
        ...

    @abstractmethod
    def dumps_weight(self) -> bytes:
        ...

    @abstractmethod
    def dump_weight(self, file: IO[bytes]) -> bytes:
        ...

    @abstractmethod
    def loads_state(self, data: bytes):
        ...

    @abstractmethod
    def load_state(self, file: IO[bytes]):
        ...

    @abstractmethod
    def dumps_state(self) -> bytes:
        ...

    @abstractmethod
    def dump_state(self, file: IO[bytes]) -> bytes:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def type(self) -> str:
        ...

    @property
    @abstractmethod
    def secure_level(self) -> int:
        ...

    @property
    @abstractmethod
    def algorithm(self) -> str:
        ...

    @property
    @abstractmethod
    def members(self) -> List[str]:
        ...

    @abstractmethod
    def run(self, node: Node):
        ...

    @abstractmethod
    def update(self, result: np.ndarray):
        ...
