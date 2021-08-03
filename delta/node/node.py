from abc import ABC, abstractmethod
from typing import IO, Any, Callable, Dict, Iterable, Optional


class Node(ABC):
    @abstractmethod
    def new_dataloader(
        self, dataset: str, dataloader: Dict[str, Any], preprocess: Callable
    ) -> Iterable:
        pass

    @abstractmethod
    def download_state(self, task_id: int) -> Optional[IO[bytes]]:
        pass

    @abstractmethod
    def upload_state(self, task_id: int, file: IO[bytes]):
        pass

    @abstractmethod
    def download_weight(self, task_id: int) -> Optional[IO[bytes]]:
        pass

    @abstractmethod
    def upload_result(self, task_id: int, data: IO[bytes]):
        pass
