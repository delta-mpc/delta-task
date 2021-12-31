from abc import ABC, abstractmethod
from typing import IO, Any, Callable, Dict, Tuple

from torch.utils.data import DataLoader


class Node(ABC):
    @abstractmethod
    def new_dataloader(
        self,
        dataset: str,
        validate_frac: float,
        cfg: Dict[str, Any],
        preprocess: Callable,
    ) -> Tuple[DataLoader, DataLoader]:
        pass

    @abstractmethod
    def download(self, type: str, dst: IO[bytes]) -> bool:
        ...

    @abstractmethod
    def upload(self, type: str, src: IO[bytes]):
        ...

    @property
    @abstractmethod
    def round(self) -> int:
        ...
