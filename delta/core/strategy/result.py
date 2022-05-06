from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch


class ResultStrategy(ABC):
    @abstractmethod
    def weight_to_params(self, weight: np.ndarray, params: Dict[str, torch.Tensor]):
        ...

    @abstractmethod
    def params_to_weight(self, params: Dict[str, torch.Tensor]) -> np.ndarray:
        ...

    @abstractmethod
    def params_to_result(
        self, params: Dict[str, torch.Tensor], last_weight: np.ndarray
    ) -> np.ndarray:
        ...

    @abstractmethod
    def result_to_params(
        self, result: np.ndarray, params: Dict[str, torch.Tensor]
    ):
        ...


class WeightResultStrategy(ResultStrategy):
    def params_to_weight(self, params: Dict[str, torch.Tensor]) -> np.ndarray:
        arrs = [p.detach().cpu().ravel().numpy() for p in params.values()]
        result = np.concatenate(arrs, axis=0)
        return result

    def weight_to_params(self, weight: np.ndarray, params: Dict[str, torch.Tensor]):
        offset = 0
        with torch.no_grad():
            for p in params.values():
                numel = p.numel()
                weight_slice = weight[offset : offset + numel]
                offset += numel
                weight_tensor = (
                    torch.from_numpy(weight_slice)
                    .to(p.dtype)
                    .to(p.device)
                    .resize_(p.shape)
                )
                p.copy_(weight_tensor)

    def params_to_result(self, params: Dict[str, torch.Tensor], last_weight: np.ndarray) -> np.ndarray:
        return self.params_to_weight(params)
    
    def result_to_params(self, result: np.ndarray, params: Dict[str, torch.Tensor]):
        return self.weight_to_params(result, params)
