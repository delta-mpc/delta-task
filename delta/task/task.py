from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, List

from ..node import Node


class Task(ABC):
    id: int

    @classmethod
    @abstractmethod
    def loads_cfg(cls, data: bytes):
        ...

    @classmethod
    @abstractmethod
    def load_cfg(cls, file: BinaryIO):
        ...

    @abstractmethod
    def dumps_cfg(self) -> bytes:
        ...

    @abstractmethod
    def dump_cfg(self, file: BinaryIO) -> bytes:
        ...

    @abstractmethod
    def loads_weight(self, data: bytes):
        ...

    @abstractmethod
    def load_weight(self, file: BinaryIO):
        ...

    @abstractmethod
    def dumps_weight(self) -> bytes:
        ...

    @abstractmethod
    def dump_weight(self, file: BinaryIO) -> bytes:
        ...

    @abstractmethod
    def loads_state(self, data: bytes):
        ...

    @abstractmethod
    def load_state(self, file: BinaryIO):
        ...

    @abstractmethod
    def dumps_state(self) -> bytes:
        ...

    @abstractmethod
    def dump_state(self, file: BinaryIO) -> bytes:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def members(self) -> Optional[List[str]]:
        ...

    @abstractmethod
    def run(self, node: Node):
        ...
