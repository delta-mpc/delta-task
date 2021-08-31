from .base import HorizontalAlgorithm
from .fed_avg import FedAvg

DefaultAlgorithm = FedAvg(
    merge_interval_epoch=0,
    merge_interval_iter=10,
    wait_timeout=60,
    connection_timeout=60,
)

__all__ = ["HorizontalAlgorithm", "FedAvg", "DefaultAlgorithm"]
