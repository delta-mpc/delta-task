from typing import List, Optional

import numpy as np
import torch

from .base import CURVE_TYPE, HorizontalAlgorithm


class FaultTolerantFedAvg(HorizontalAlgorithm):
    def __init__(
        self,
        merge_interval_iter: int = 0,
        merge_interval_epoch: int = 1,
        min_clients: int = 2,
        max_clients: int = 2,
        wait_timeout: Optional[float] = None,
        connection_timeout: Optional[float] = None,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
    ):
        super().__init__(
            "FaultTolerantFedAvg",
            merge_interval_iter=merge_interval_iter,
            merge_interval_epoch=merge_interval_epoch,
            min_clients=min_clients,
            max_clients=max_clients,
            wait_timeout=wait_timeout,
            connection_timeout=connection_timeout,
            fault_tolerant=True,
            precision=precision,
            curve=curve,
        )

    def params_to_result(self, params: List[torch.Tensor]) -> np.ndarray:
        return self.params_to_weight(params)

    def params_to_weight(self, params: List[torch.Tensor]) -> np.ndarray:
        arrs = [p.detach().cpu().ravel().numpy() for p in params]
        result = np.concatenate(arrs, axis=0)
        return result

    def weight_to_params(self, weight: np.ndarray, params: List[torch.Tensor]):
        offset = 0
        with torch.no_grad():
            for p in params:
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
