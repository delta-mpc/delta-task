from __future__ import annotations

from abc import abstractmethod
from typing import List, Tuple, Dict

from ..core.strategy import AnalyticsStrategy, CURVE_TYPE
from ..core.task import GraphNode, InputGraphNode
from .task import HorizontalTask
import delta.dataset


class HorizontalAnalytics(HorizontalTask):
    def __init__(
        self,
        name: str,
        min_clients: int = 2,
        max_clients: int = 2,
        wait_timeout: float = 60,
        connection_timeout: float = 60,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
    ) -> None:
        strategy = AnalyticsStrategy(
            min_clients=min_clients,
            max_clients=max_clients,
            wait_timeout=wait_timeout,
            connection_timeout=connection_timeout,
            precision=precision,
            curve=curve,
        )
        super().__init__(name, strategy)

    @abstractmethod
    def execute(self, **inputs: InputGraphNode) -> GraphNode | Tuple[GraphNode]:
        ...

    @abstractmethod
    def dataset(self) -> Dict[str, delta.dataset.DataFrame]:
        ...

    def _build_graph(self) -> Tuple[List[InputGraphNode], List[GraphNode]]:
        inputs = self.dataset()
        for name, node in inputs.items():
            node.name = name
        outputs = self.execute(**inputs)
        if isinstance(outputs, tuple):
            return list(inputs.values()), list(outputs)
        else:
            return list(inputs.values()), [outputs]
