from .base import HorizontalAlgorithm
from .fault_tolerant_fedavg import FaultTolerantFedAvg
from .fedavg import FedAvg

DefaultAlgorithm = FedAvg(
    merge_interval_epoch=0,
    merge_interval_iter=10,
    wait_timeout=60,
    connection_timeout=60,
)

__all__ = ["HorizontalAlgorithm", "FedAvg", "DefaultAlgorithm", "FaultTolerantFedAvg"]
