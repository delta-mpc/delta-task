import json
import logging
from abc import abstractmethod
from tempfile import TemporaryFile
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from delta import serialize
from delta.algorithm.horizontal import DefaultAlgorithm, HorizontalAlgorithm
from delta.node import Node

from .task import Task

_logger = logging.getLogger(__name__)


class LoadError(Exception):
    def __init__(self, type: str) -> None:
        self.type = type

    def __str__(self) -> str:
        return f"load {type} failed"


class HorizontalTask(Task):
    def __init__(
        self,
        name: str,
        dataset: str,
        max_rounds: int,
        validate_interval: int = 1,
        validate_frac: float = 0.1,
    ):
        super().__init__(name, dataset)
        self.max_rounds = max_rounds
        self.validate_interval = validate_interval
        self.validate_frac = validate_frac

        self._alg = self.algorithm()

        self._state = {
            "epoch": 1,
            "iteration": 1,
        }

    @property
    def type(self) -> str:
        return "horizontal"

    @property
    def epoch(self) -> int:
        return self._state["epoch"]

    @epoch.setter
    def epoch(self, epoch: int):
        self._state["epoch"] = epoch

    @property
    def iteration(self) -> int:
        return self._state["iteration"]

    @iteration.setter
    def iteration(self, iteration: int):
        self._state["iteration"] = iteration

    @abstractmethod
    def train(self, dataloader: Iterable):
        ...

    @abstractmethod
    def get_params(self) -> List[torch.Tensor]:
        ...

    @abstractmethod
    def validate(self, dataloader: Iterable) -> Dict[str, float]:
        ...

    @abstractmethod
    def preprocess(self, x, y=None):
        ...

    def dataloader_config(
        self,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        return {"shuffle": True, "batch_size": 64, "drop_last": True}

    def algorithm(self) -> HorizontalAlgorithm:
        return DefaultAlgorithm

    def _load_state(self, node: Node):
        with TemporaryFile("w+b") as f:
            if node.download("state", f):
                f.seek(0)
                self._state = torch.load(f)
            else:
                raise LoadError("state")

    def _save_state(self, node: Node):
        with TemporaryFile("w+b") as f:
            torch.save(self._state, f)
            f.seek(0)
            node.upload("state", f)

    def _load_weight(self, node: Node):
        with TemporaryFile("w+b") as f:
            if node.download("weight", f):
                f.seek(0)
                weight = serialize.load_arr(f)
                self._alg.weight_to_params(weight, self.get_params())
            else:
                raise LoadError("weight")

    def _upload_result(self, node: Node):
        with TemporaryFile("w+b") as f:
            arr = self._alg.params_to_result(self.get_params())
            serialize.dump_arr(f, arr)
            f.seek(0)
            node.upload("result", f)

    def _upload_metrics(self, node: Node, metrics: Dict[str, float]):
        with TemporaryFile("w+b") as f:
            s = json.dumps(metrics).encode("utf-8")
            f.write(s)
            f.seek(0)
            node.upload("metrics", f)

    def get_weight(self) -> np.ndarray:
        return self._alg.params_to_weight(self.get_params())

    def run(self, node: Node):
        try:
            cfg = self.dataloader_config()
            if isinstance(cfg, tuple):
                train_loader_cfg, val_loader_cfg = cfg
                dataloader_cfg = {"train": train_loader_cfg, "validate": val_loader_cfg}
            else:
                dataloader_cfg = {"train": cfg, "validate": cfg}
            train_loader, val_loader = node.new_dataloader(
                self.dataset, self.validate_frac, dataloader_cfg, self.preprocess
            )

            # download start and weight
            try:
                self._load_state(node)
            except LoadError:
                _logger.info("no initial state")
            try:
                self._load_weight(node)
            except LoadError:
                _logger.info("no initial weight")
            _logger.info(
                f"train start from epoch {self.epoch} iteration {self.iteration}"
            )

            def train_context():
                _logger.info(f"start round {node.round}")
                finished = False

                while not finished:
                    for batch in train_loader:
                        if finished:
                            break
                        
                        _logger.info(f"epoch {self.epoch} iteration {self.iteration}")

                        yield batch

                        if self._alg.should_merge(self.epoch, self.iteration, False):
                            _logger.info(f"iteration {self.iteration}, start to merge")
                            self._upload_result(node)
                            finished = True
                        self.iteration += 1

                    if self._alg.should_merge(self.epoch, self.iteration, True):
                        _logger.info(f"epoch {self.epoch}, start to merge")
                        self._upload_result(node)
                        finished = True
                    if (self.iteration - 1) % len(train_loader) == 0:
                        self.epoch += 1

                if node.round % self.validate_interval == 0:
                    _logger.info(f"round {node.round} start to validate")
                    metrics = self.validate(val_loader)
                    _logger.info(f"metrics: {metrics}")
                    self._upload_metrics(node, metrics)

                self._save_state(node)
                _logger.info(f"training round {node.round} finished")

            self.train(train_context())

        except Exception as e:
            _logger.exception(e)
            raise
