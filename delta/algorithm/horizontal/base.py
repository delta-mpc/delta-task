from abc import ABC, abstractmethod
from typing import List, Literal, Optional, IO

import numpy as np
import torch


CURVE_TYPE = Literal[
    "secp192r1",
    "secp224r1",
    "secp256k1",
    "secp256r1",
    "secp384r1",
    "secp521r1"
]


class HorizontalAlgorithm(ABC):
    def __init__(
        self,
        name: str,
        merge_interval_iter: int = 0,
        merge_interval_epoch: int = 1,
        min_clients: int = 2,
        max_clients: int = 100,
        wait_timeout: Optional[float] = None,
        connection_timeout: Optional[float] = None,
        fault_tolerant: bool = False,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1"
    ):
        assert (
            merge_interval_epoch * merge_interval_iter == 0
        ), "one of merge_interval_epoch and merge_interval_iter should be zero"
        assert (
            merge_interval_epoch + merge_interval_iter > 0
        ), "merge_interval_epoch and merge_interval_iter should not all be zero"
        self.name = name
        self.merge_interval_iter = merge_interval_iter
        self.merge_interval_epoch = merge_interval_epoch
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.wait_timeout = wait_timeout
        self.connnection_timeout = connection_timeout
        self.fault_tolerant = fault_tolerant
        self.precision = precision
        self.curve = curve

    def should_merge(self, epoch: int, iteration: int, epoch_finished: bool):
        if epoch_finished:
            if self.merge_interval_epoch > 0 and epoch % self.merge_interval_epoch == 0:
                return True
        if not epoch_finished:
            if (
                self.merge_interval_iter > 0
                and iteration % self.merge_interval_iter == 0
            ):
                return True
        return False

    @abstractmethod
    def params_to_result(self, params: List[torch.Tensor]) -> np.ndarray:
        ...

    @abstractmethod
    def params_to_weight(self, params: List[torch.Tensor]) -> np.ndarray:
        ...

    @abstractmethod
    def weight_to_params(self, weight: np.ndarray, params: List[torch.Tensor]):
        ...
