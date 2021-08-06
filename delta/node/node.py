from abc import ABC, abstractmethod
from typing import IO, Any, Callable, Dict, Iterable, Optional


class Node(ABC):
    @abstractmethod
    def new_dataloader(
        self, dataset: str, dataloader: Dict[str, Any], preprocess: Callable
    ) -> Iterable:
        pass

    @abstractmethod
    def download_state(self, dst: IO[bytes]) -> bool:
        pass

    @abstractmethod
    def upload_state(self, file: IO[bytes]):
        pass

    @abstractmethod
    def download_weight(self, dst: IO[bytes]) -> bool:
        pass

    @abstractmethod
    def upload_result(self, data: IO[bytes]):
        pass

    @abstractmethod
    def finish(self):
        pass
