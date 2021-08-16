import logging
import os.path
import shutil
from tempfile import TemporaryDirectory
from typing import IO, Any, Callable, Dict, Iterable

from ..data import new_dataloader
from .node import Node


class DebugNode(Node):
    def __init__(self, task_id: int):
        self._logger = logging.getLogger(__name__)

        self._temp_dir = TemporaryDirectory()

        self._task_id = task_id
        self._state_count = 0
        self._weight_count = 0

    def state_file(self) -> str:
        return os.path.join(
            self._temp_dir.name, f"{self._task_id}.state.{self._state_count}"
        )

    def weight_file(self) -> str:
        return os.path.join(
            self._temp_dir.name, f"{self._task_id}.weight.{self._weight_count}"
        )

    def new_dataloader(
        self, dataset: str, dataloader: Dict[str, Any], preprocess: Callable
    ) -> Iterable:
        return new_dataloader(dataset, dataloader, preprocess)

    def download_state(self, dst: IO[bytes]) -> bool:
        filename = self.state_file()
        if os.path.exists(filename):
            self._logger.info(f"load state {filename} for task {self._task_id}")
            with open(filename, mode="rb") as f:
                shutil.copyfileobj(f, dst)
            return True
        else:
            self._logger.info(f"initial state for task {self._task_id}")
            return False

    def upload_state(self, file: IO[bytes]):
        self._state_count += 1
        filename = self.state_file()
        with open(filename, mode="wb") as f:
            shutil.copyfileobj(file, f)
        self._logger.info(f"dump state {filename} for task {self._task_id}")

    def upload_result(self, file: IO[bytes]):
        self._weight_count += 1
        filename = self.weight_file()
        with open(filename, mode="wb") as f:
            shutil.copyfileobj(file, f)
        self._logger.info(f"upload weight {filename} for task {self._task_id}")

    def download_weight(self, dst: IO[bytes]) -> bool:
        filename = self.weight_file()
        if os.path.exists(filename):
            self._logger.info(f"download weight {filename} for task {self._task_id}")
            with open(filename, mode="rb") as f:
                shutil.copyfileobj(f, dst)
            return True
        else:
            self._logger.info(f"initial weight for task {self._task_id}")
            return False

    def finish(self):
        self._logger.info(f"task {self._task_id} finished")
