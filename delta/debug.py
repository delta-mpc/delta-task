from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .dataset import load_dataframe, load_dataset
from .core.task import (ClientContext, DataFormat, DataLocation, DataNode,
                        InputGraphNode, ServerContext, Task)


class Aggregator(object):
    def __init__(self) -> None:
        self._data = {}

    def upload(self, name: str, data: Dict[str, Any]):
        self._data[name] = data

    def gather(self, name: str) -> Tuple[Dict[str, Any], int]:
        return self._data[name], 1


class DebugClientContext(ClientContext):
    def __init__(self, aggregator: Aggregator, server_ctx: ServerContext) -> None:
        self._state = {}
        self._aggregator = aggregator
        self._server_ctx = server_ctx

    def get(self, *vars: DataNode) -> List[Any]:
        res = []
        for var in vars:
            value = None
            if var.location == DataLocation.CLIENT:
                value = self._state.get(var.name)
                if value is None and isinstance(var, InputGraphNode):
                    if var.filename is not None and var.format is not None:
                        value = self.load_data(var.filename, var.format, **var.kwargs)
                    elif var.default is not None:
                        value = var.default

                    if value is not None:
                        self._state[var.name] = value
            else:
                value = self._server_ctx.get(var)[0]

            if value is None:
                raise ValueError(f"Cannot get var {var.name}")

            res.append(value)

        return res

    def set(self, *pairs: Tuple[DataNode, Any]):
        for var, data in pairs:
            self._state[var.name] = data

    def has(self, key: str) -> bool:
        return key in self._state

    def upload(self, name: str, data: Dict[str, Any]):
        self._aggregator.upload(name, data)

    def load_data(self, filename: str, format: DataFormat, **kwargs):
        if format == DataFormat.DATASET:
            return load_dataset(filename, **kwargs)
        elif format == DataFormat.DATAFRAME:
            return load_dataframe(filename)

    def clear(self):
        self._state.clear()


class DebugServerContext(ServerContext):
    def __init__(self, aggregator: Aggregator) -> None:
        self._state = {}
        self._aggregator = aggregator

    def get(self, *vars: DataNode) -> List[Any]:
        res = []
        for var in vars:
            value = self._state.get(var.name)
            if value is None and isinstance(var, InputGraphNode):
                if var.default is not None:
                    value = var.default
                    self._state[var.name] = value

            if value is None:
                raise ValueError(f"Cannot get var {var.name}")

            res.append(value)

        return res

    def set(self, *pairs: Tuple[DataNode, Any]):
        for var, data in pairs:
            self._state[var.name] = data

    def has(self, key: str) -> bool:
        return key in self._state

    def gather(self, name: str) -> Tuple[Dict[str, Any], int]:
        return self._aggregator.gather(name)

    def clear(self):
        self._state.clear()


def debug(task: Task, **data: Any):
    aggregator = Aggregator()
    server_context = DebugServerContext(aggregator)
    client_context = DebugClientContext(aggregator, server_context)

    for var in task.inputs:
        if var.name in data:
            if var.location == DataLocation.CLIENT:
                client_context.set((var, data[var.name]))
            elif var.location == DataLocation.SERVER:
                server_context.set((var, data[var.name]))

    for step in task.steps:
        step.map(client_context)
        step.reduce(server_context)

    res = tuple(server_context.get(*task.outputs))
    if len(res) == 1:
        return res[0]
    else:
        return res
