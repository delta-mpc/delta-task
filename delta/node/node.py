from abc import ABC, abstractmethod
from typing import IO, Any, Callable, Dict, Iterable, Optional, Tuple


class Node(ABC):
    @abstractmethod
    def new_dataloader(
        self, dataset: str, validate_frac: float, cfg: Dict[str, Any], preprocess: Callable
    ) -> Tuple[Iterable, Iterable]:
        pass

    @abstractmethod
    def download(self, type: str, dst: IO[bytes]) -> bool:
        ...

    @abstractmethod
    def upload(self, type: str, src: IO[bytes]):
        ...

    @abstractmethod
    def finish(self):
        pass
