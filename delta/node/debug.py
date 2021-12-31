import logging
import os.path
import shutil
from typing import IO, Any, Callable, Dict, Iterable, Tuple

from ..data import new_dataloader
from .node import Node


class DebugNode(Node):
    def __init__(self, task_id: str, round: int, dirname: str):
        self._logger = logging.getLogger(__name__)

        self._dir = dirname

        self._task_id = task_id
        self._round = round

    @property
    def round(self) -> int:
        return self._round

    def state_file(self, round: int) -> str:
        return os.path.join(self._dir, f"{self._task_id}.{round}.state")

    def weight_file(self, round: int) -> str:
        return os.path.join(self._dir, f"{self._task_id}.{round}.weight")

    def metrics_file(self, round: int) -> str:
        return os.path.join(self._dir, f"{self._task_id}.{round}.metrics")

    def new_dataloader(
        self,
        dataset: str,
        validate_frac: float,
        cfg: Dict[str, Any],
        preprocess: Callable,
    ) -> Tuple[Iterable, Iterable]:
        assert "train" in cfg, "cfg should contain train"
        assert "validate" in cfg, "cfg should contain validate"

        train_loader = new_dataloader(dataset, cfg["train"], preprocess)
        val_loader = new_dataloader(dataset, cfg["validate"], preprocess)
        return train_loader, val_loader

    def download(self, type: str, dst: IO[bytes]) -> bool:
        if type == "state":
            return self.download_state(dst)
        elif type == "weight":
            return self.download_weight(dst)
        raise ValueError(f"unknown download type {type}")

    def upload(self, type: str, src: IO[bytes]):
        if type == "state":
            return self.upload_state(src)
        elif type == "result":
            return self.upload_result(src)
        elif type == "metrics":
            return self.upload_metrics(src)
        raise ValueError(f"unknown upload type {type}")

    def download_state(self, dst: IO[bytes]) -> bool:
        filename = self.state_file(self.round - 1)
        if os.path.exists(filename):
            self._logger.info(f"load state {filename} for task {self._task_id}")
            with open(filename, mode="rb") as f:
                shutil.copyfileobj(f, dst)
            return True
        else:
            self._logger.info(f"initial state for task {self._task_id}")
            return False

    def upload_state(self, file: IO[bytes]):
        filename = self.state_file(self.round)
        with open(filename, mode="wb") as f:
            shutil.copyfileobj(file, f)
        self._logger.info(f"dump state {filename} for task {self._task_id}")

    def upload_result(self, file: IO[bytes]):
        filename = self.weight_file(self.round)
        with open(filename, mode="wb") as f:
            shutil.copyfileobj(file, f)
        self._logger.info(f"upload weight {filename} for task {self._task_id}")

    def upload_metrics(self, file: IO[bytes]):
        filename = self.metrics_file(self.round)
        with open(filename, mode="wb") as f:
            shutil.copyfileobj(file, f)
        self._logger.info(f"upload metrics {filename} for task {self._task_id}")

    def download_weight(self, dst: IO[bytes]) -> bool:
        filename = self.weight_file(self.round - 1)
        if os.path.exists(filename):
            self._logger.info(f"download weight {filename} for task {self._task_id}")
            with open(filename, mode="rb") as f:
                shutil.copyfileobj(f, dst)
            return True
        else:
            self._logger.info(f"initial weight for task {self._task_id}")
            return False
