from __future__ import annotations

import abc
from typing import Any, Dict, List, Tuple

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
    TYPE: TaskType = TaskType.HORIZONTAL

    def __init__(
        self, name: str, strategy: Strategy, enable_verify: bool = False
    ) -> None:
        self.name = name
        self.strategy = strategy
        self.enable_verify = enable_verify

    @abc.abstractmethod
    def dataset(self) -> Dict[str, InputGraphNode] | InputGraphNode:
        ...

    @abc.abstractmethod
    def _build_graph(self) -> Tuple[List[InputGraphNode], List[GraphNode]]:
        ...

    def options(self) -> Dict[str, Any]:
        return {}

    def build(self) -> Task:
        inputs, outputs = self._build_graph()
        constructor = TaskConstructer(
            name=self.name,
            inputs=inputs,
            outputs=outputs,
            strategy=self.strategy,
            type=self.TYPE,
            enable_verify=self.enable_verify,
            options=self.options(),
        )
        return build(constructor)
