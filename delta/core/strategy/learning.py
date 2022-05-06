from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .merge import EpochMergeStrategy, MergeStrategy
from .result import ResultStrategy, WeightResultStrategy
from .select import RandomSelectStrategy, SelectStrategy
from .strategy import CURVE_TYPE, Strategy


class LearningStrategy(Strategy):
    def __init__(
        self,
        name: str,
        select_strategy: SelectStrategy,
        merge_strategy: MergeStrategy,
        result_strategy: ResultStrategy,
        wait_timeout: float = 60,
        connection_timeout: float = 60,
        fault_tolerant: bool = False,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
    ) -> None:
        super().__init__(
            name=name,
            select_strategy=select_strategy,
            wait_timeout=wait_timeout,
            connection_timeout=connection_timeout,
            fault_tolerant=fault_tolerant,
            precision=precision,
            curve=curve,
        )
        self.merge_strategy = merge_strategy
        self.result_strategy = result_strategy

    def should_merge(self, epoch: int, iteration: int, epoch_finished: bool) -> bool:
        return self.merge_strategy.should_merge(epoch, iteration, epoch_finished)

    def params_to_weight(self, params: Dict[str, torch.Tensor]) -> np.ndarray:
        return self.result_strategy.params_to_weight(params)

    def weight_to_params(self, weight: np.ndarray, params: Dict[str, torch.Tensor]):
        return self.result_strategy.weight_to_params(weight, params)

    def params_to_result(
        self, params: Dict[str, torch.Tensor], last_weight: np.ndarray
    ) -> np.ndarray:
        return self.result_strategy.params_to_result(params, last_weight)

    def result_to_params(self, result: np.ndarray, params: Dict[str, torch.Tensor]):
        return self.result_strategy.result_to_params(result, params)
