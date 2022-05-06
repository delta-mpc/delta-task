from __future__ import annotations

import abc
from typing import Dict, List, Tuple

from ..core.strategy import Strategy
from ..core.task import (
    GraphNode,
    InputGraphNode,
    Task,
    TaskConstructer,
    TaskType,
    build,
)


class HorizontalTask(abc.ABC):
    def __init__(self, name: str, strategy: Strategy) -> None:
        self.name = name
        self.strategy = strategy

    @abc.abstractmethod
    def dataset(self) -> Dict[str, InputGraphNode] | InputGraphNode:
        ...

    @abc.abstractmethod
    def _build_graph(self) -> Tuple[List[InputGraphNode], List[GraphNode]]:
        ...

    def build(self) -> Task:
        inputs, outputs = self._build_graph()
        constructor = TaskConstructer(
            name=self.name,
            inputs=inputs,
            outputs=outputs,
            strategy=self.strategy,
            type=TaskType.HORIZONTAL,
        )
        return build(constructor)
