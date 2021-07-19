from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, Callable, Optional


class Node(ABC):
    @abstractmethod
    def new_dataloader(self, dataset: str, dataloader: Dict[str, Any], preprocess: Callable) -> Iterable:
        pass

    @abstractmethod
    def download_state(self, task_id: int) -> Optional[bytes]:
        pass

    @abstractmethod
    def upload_state(self, task_id: int, data: bytes):
        pass

    @abstractmethod
    def download_weight(self, task_id: int) -> Optional[bytes]:
        pass

    @abstractmethod
    def upload_weight(self, task_id: int, data: bytes):
        pass
