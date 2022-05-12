from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas

from .strategy import Strategy

__all__ = [
    "DataNode",
    "OpType",
    "GraphNode",
    "Operator",
    "OperatorGroup",
    "MapOperator",
    "MapOperatorGroup",
    "MapReduceOperator",
    "MapReduceOperatorGroup",
    "ReduceOperator",
    "ReduceOperatorGroup",
    "Context",
    "ClientContext",
    "ServerContext",
    "Step",
    "Task",
    "build",
    "DataLocation",
    "DataFormat",
    "InputGraphNode",
    "TaskConstructer",
    "AggValueType",
    "AggResultType",
]


class DataLocation(str, Enum):
    CLIENT = "CLIENT"
    SERVER = "SERVER"


class DataFormat(str, Enum):
    DATASET = "DATASET"
    DATAFRAME = "DATAFRAME"
    SERIES = "SERIES"


class DataNode(object):
    def __init__(
        self,
        name: str = "",
        location: DataLocation = DataLocation.CLIENT,
    ) -> None:
        self.name = name
        self.location = location

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class OpType(str, Enum):
    MAP = "MAP"
    MAP_REDUCE = "MAP_REDUCE"
    REDUCE = "REDUCE"


class Operator(abc.ABC):
    priority: int = 100

    def __init__(
        self, name: str, inputs: List[DataNode], outputs: List[DataNode], **kwargs: Any
    ) -> None:
        """
        A single operator

        name: operator name
        inputs: operator input variables
        outputs: operator output variables
        kwargs: key word arguments for the operator implementation
        """
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.kwargs = kwargs

    @property
    @abc.abstractmethod
    def type(self) -> OpType:
        ...

    def __repr__(self) -> str:
        return f"{self.name}(type={self.type}, inputs={self.inputs}, outputs={self.outputs}, kwargs={self.kwargs})"

    def __str__(self) -> str:
        return f"""
            {self.name}
            (
                type={self.type}, 
                inputs={self.inputs}, 
                outputs={self.outputs}, 
                kwargs={self.kwargs}
            )
            """

    def __lt__(self, other: "Operator"):
        return self.priority < other.priority


class OperatorGroup(object):
    def __init__(self, *ops: Operator) -> None:
        """
        A group of operators which are independent with each other.
        Can execute these operators parallelly.

        ops: operators
        """
        self.ops = sorted(ops)

        input_set = set()
        output_set = set()
        for op in ops:
            input_set = input_set.union(op.inputs)
            output_set = output_set.union(op.outputs)
        self.name = "group"
        self.inputs = list(input_set)
        self.outputs = list(output_set)

    def __repr__(self) -> str:
        return (
            f"{self.name}(inputs={self.inputs}, ouputs={self.outputs}, ops={self.ops})"
        )

    def __str__(self) -> str:
        return f"""
            {self.name}
            (
                inputs={self.inputs}, 
                ouputs={self.outputs}, 
                ops={self.ops}
            )
            """


class GraphNode(DataNode):
    def __init__(
        self,
        name: str = "",
        location: DataLocation = DataLocation.CLIENT,
        src: Operator | None = None,
    ) -> None:
        super().__init__(name, location)
        self.src = src


class InputGraphNode(GraphNode):
    def __init__(
        self,
        name: str = "",
        location: DataLocation = DataLocation.CLIENT,
        src: Operator | None = None,
        format: DataFormat | None = None,
        filename: str | None = None,
        default: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, location, src)
        self.format = format
        self.filename = filename
        self.default = default
        self.kwargs = kwargs


class MapOperator(Operator, metaclass=abc.ABCMeta):
    @property
    def type(self) -> OpType:
        return OpType.MAP

    @abc.abstractmethod
    def map(self, *args, **kwargs) -> Any:
        ...


class MapOperatorGroup(OperatorGroup):
    def __init__(self, *ops: MapOperator) -> None:
        assert all(op.type == OpType.MAP for op in ops)
        super().__init__(*ops)
        self.name = "map_group"

    @property
    def type(self) -> OpType:
        return OpType.MAP


class ReduceOperator(Operator, metaclass=abc.ABCMeta):
    @property
    def type(self) -> OpType:
        return OpType.REDUCE

    @abc.abstractmethod
    def reduce(self, *args, **kwargs) -> Any:
        ...


class ReduceOperatorGroup(OperatorGroup):
    def __init__(self, *ops: ReduceOperator) -> None:
        assert all(op.type == OpType.REDUCE for op in ops)
        super().__init__(*ops)
        self.name = "reduce_group"

    @property
    def type(self) -> OpType:
        return OpType.REDUCE


AggValueType = Union[np.ndarray, pandas.DataFrame, pandas.Series]

AggResultType = Dict[str, AggValueType]


class MapReduceOperator(Operator, metaclass=abc.ABCMeta):
    def __init__(
        self,
        name: str,
        map_inputs: List[DataNode],
        reduce_inputs: List[DataNode],
        map_outputs: List[DataNode],
        reduce_outputs: List[DataNode],
        **kwargs: Any,
    ) -> None:
        inputs = list(set(map_inputs) | set(reduce_inputs))
        outputs = list(set(map_outputs) | set(reduce_outputs))
        super().__init__(name, inputs, outputs, **kwargs)
        self.map_inputs = map_inputs
        self.reduce_inputs = reduce_inputs
        self.map_outputs = map_outputs
        self.reduce_outputs = reduce_outputs

    @property
    def agg_name(self) -> str:
        return "agg({})".format(",".join(var.name for var in self.reduce_outputs))

    @property
    def type(self) -> OpType:
        return OpType.MAP_REDUCE

    @abc.abstractmethod
    def map(self, *args, **kwargs) -> AggResultType | Tuple[AggResultType, ...]:
        ...

    @abc.abstractmethod
    def reduce(self, data: AggResultType, node_count: int, *args: Any) -> Any:
        ...


class MapReduceOperatorGroup(OperatorGroup):
    def __init__(self, *ops: MapReduceOperator) -> None:
        assert all(op.type == OpType.MAP_REDUCE for op in ops)
        super().__init__(*ops)
        self.name = "map_reduce_group"
        self.ops = ops

    @property
    def agg_names(self) -> List[str]:
        return [op.agg_name for op in self.ops]

    @property
    def type(self) -> OpType:
        return OpType.MAP_REDUCE


class Context(abc.ABC):
    @abc.abstractmethod
    def get(self, *vars: DataNode) -> List[Any]:
        ...

    @abc.abstractmethod
    def set(self, *pairs: Tuple[DataNode, Any]):
        ...

    @abc.abstractmethod
    def has(self, var: DataNode) -> bool:
        ...

    @abc.abstractmethod
    def execute(self, group: OperatorGroup):
        ...

    @abc.abstractmethod
    def clear(self):
        ...


def _get_inputs(ctx: Context, inputs: List[DataNode]) -> List[Any]:
    return ctx.get(*inputs)


def _set_outputs(ctx: Context, outputs: List[DataNode], result: Any):
    if len(outputs) == 1:
        result = (result,)
    assert len(outputs) == len(result), "result length is not equal to outputs length"
    ctx.set(*zip(outputs, result))


class ClientContext(Context):
    @abc.abstractmethod
    def upload(self, name: str, data: Dict[str, Any]):
        ...

    def _execute_op(self, op: Operator):
        if isinstance(op, ReduceOperator):
            raise TypeError("cannot execute reduce operator on client side")

        elif isinstance(op, MapOperator):
            args = _get_inputs(self, op.inputs)
            res = op.map(*args)
            _set_outputs(self, op.outputs, res)

        elif isinstance(op, MapReduceOperator):
            args = _get_inputs(self, op.map_inputs)
            res = op.map(*args)
            if isinstance(res, tuple):
                agg_result = res[0]
                vars = res[1:]
                _set_outputs(self, op.map_outputs, vars)
            else:
                agg_result = res
            self.upload(op.agg_name, agg_result)

    def execute(self, group: OperatorGroup):
        if isinstance(group, ReduceOperatorGroup):
            raise TypeError("cannot execute reduce operator on client side")
        for op in group.ops:
            self._execute_op(op)


class ServerContext(Context):
    @abc.abstractmethod
    def gather(self, name: str) -> Tuple[Dict[str, Any], int]:
        ...

    def _execute_op(self, op: Operator):
        if isinstance(op, MapOperator):
            raise TypeError("cannot execute map operator on server side")

        elif isinstance(op, MapReduceOperator):
            data, node_count = self.gather(op.agg_name)
            args = _get_inputs(self, op.reduce_inputs)
            res = op.reduce(data, node_count, *args)
            _set_outputs(self, op.reduce_outputs, res)

        elif isinstance(op, ReduceOperator):
            args = _get_inputs(self, op.inputs)
            res = op.reduce(*args)
            _set_outputs(self, op.outputs, res)

    def execute(self, group: OperatorGroup):
        if isinstance(group, MapOperatorGroup):
            raise TypeError("cannot execute map operator on server side")
        for op in group.ops:
            self._execute_op(op)


@dataclass
class Step(object):
    map_ops: List[MapOperatorGroup] = field(default_factory=list)
    map_reduce_ops: List[MapReduceOperatorGroup] = field(default_factory=list)
    reduce_ops: List[ReduceOperatorGroup] = field(default_factory=list)

    _inputs: List[DataNode] | None = field(init=False, default=None)
    _outputs: List[DataNode] | None = field(init=False, default=None)
    _agg_names: List[str] | None = field(init=False, default=None)

    @property
    def inputs(self) -> List[DataNode]:
        if self._inputs is None:
            inputs = set()
            for layer in [self.map_ops, self.map_reduce_ops, self.reduce_ops]:
                for group in layer:
                    inputs = inputs.union(group.inputs)
            outputs = set()
            for layer in [self.map_ops, self.map_reduce_ops, self.reduce_ops]:
                for group in layer:
                    outputs = outputs.union(group.outputs)
            self._inputs = list(inputs - outputs)
            return self._inputs
        else:
            return self._inputs

    @property
    def outputs(self) -> List[DataNode]:
        if self._outputs is None:
            inputs = set()
            for layer in [self.map_ops, self.map_reduce_ops, self.reduce_ops]:
                for group in layer:
                    inputs = inputs.union(group.inputs)
            outputs = set()
            for layer in [self.map_ops, self.map_reduce_ops, self.reduce_ops]:
                for group in layer:
                    outputs = outputs.union(group.outputs)
            self._outputs = list(outputs - inputs)
            return self._outputs
        else:
            return self._outputs

    @property
    def agg_names(self) -> List[str]:
        if self._agg_names is None:
            self._agg_names = []
            for op_group in self.map_reduce_ops:
                self._agg_names.extend(op_group.agg_names)
            return self._agg_names
        else:
            return self._agg_names

    def map(self, ctx: ClientContext):
        # map
        for group in self.map_ops:
            ctx.execute(group)

        # map of map reduce
        for group in self.map_reduce_ops:
            ctx.execute(group)

    def reduce(self, ctx: ServerContext):
        # reduce of map reduce
        for group in self.map_reduce_ops:
            ctx.execute(group)

        # reduce
        for group in self.reduce_ops:
            ctx.execute(group)


class TaskType(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class Task(object):
    name: str
    dataset: List[str]
    type: TaskType
    steps: List[Step]
    inputs: List[InputGraphNode]
    outputs: List[DataNode]
    strategy: Strategy


@dataclass
class TaskConstructer(object):
    name: str
    inputs: List[InputGraphNode]
    outputs: List[GraphNode]
    strategy: Strategy
    type: TaskType


def build(constructor: TaskConstructer) -> Task:
    graph = nx.DiGraph()
    # bfs build graph
    queue: List[GraphNode | Operator] = list(constructor.outputs)
    while len(queue):
        next_queue = []
        for node in queue:
            if isinstance(node, GraphNode):
                if node.src:
                    graph.add_edge(node.src, node)
                    next_queue.append(node.src)
            elif isinstance(node, Operator):
                for src in node.inputs:
                    graph.add_edge(src, node)
                    next_queue.append(src)
        queue = next_queue

    # name variables
    var_count = 0
    for node in nx.topological_sort(graph):
        if isinstance(node, GraphNode):
            if len(node.name) == 0:
                node.name = f"_{var_count}"
            var_count += 1

    # sort operators
    op_layers = []
    inputs = constructor.inputs
    init_ready = False
    for layer in nx.topological_generations(graph):
        if isinstance(layer[0], Operator):
            op_layers.append(layer)
        elif not init_ready and isinstance(layer[0], GraphNode):
            assert all(
                isinstance(node, InputGraphNode) for node in layer
            ), "All input variables should be instance of InputGraphNode"
            assert (
                len(set(inputs) - set(layer)) == 0
            ), "Graph first layer doesn't include all inputs"
            inputs = layer
            init_ready = True

    dataset = []
    for node in inputs:
        if (
            isinstance(node, InputGraphNode)
            and node.filename is not None
            and node.format is not None
        ):
            dataset.append(node.filename)

    # construct operator matrix
    step_layers: List[
        Tuple[
            MapOperatorGroup | None,
            MapReduceOperatorGroup | None,
            ReduceOperatorGroup | None,
        ]
    ] = []
    for layer in op_layers:
        map_ops: List[MapOperator] = []
        map_reduce_ops: List[MapReduceOperator] = []
        reduce_ops: List[ReduceOperator] = []
        for op in layer:
            if isinstance(op, MapOperator):
                map_ops.append(op)
            elif isinstance(op, MapReduceOperator):
                map_reduce_ops.append(op)
            elif isinstance(op, ReduceOperator):
                reduce_ops.append(op)
        if len(map_ops) > 0:
            map_group = MapOperatorGroup(*map_ops)
        else:
            map_group = None
        if len(map_reduce_ops) > 0:
            map_reduce_group = MapReduceOperatorGroup(*map_reduce_ops)
        else:
            map_reduce_group = None
        if len(reduce_ops) > 0:
            reduce_group = ReduceOperatorGroup(*reduce_ops)
        else:
            reduce_group = None
        step_layers.append((map_group, map_reduce_group, reduce_group))

    # greedy algorithm to generate steps
    ready_vars = set(inputs)
    steps = []
    map_i, map_reduce_i, reduce_i = 0, 0, 0
    layer_count = len(step_layers)
    while map_i < layer_count or map_reduce_i < layer_count or reduce_i < layer_count:
        step = Step()
        while map_i < layer_count:
            op = step_layers[map_i][0]
            if op is None:
                map_i += 1
            elif len(set(op.inputs) & ready_vars) == len(op.inputs):
                step.map_ops.append(op)
                ready_vars = ready_vars.union(op.outputs)
                map_i += 1
            else:
                break
        map_reduce_outputs = []
        while map_reduce_i < layer_count:
            op = step_layers[map_reduce_i][1]
            if op is None:
                map_reduce_i += 1
            elif len(set(op.inputs) & ready_vars) == len(op.inputs):
                step.map_reduce_ops.append(op)
                map_reduce_outputs.extend(op.outputs)
                map_reduce_i += 1
            else:
                break
        ready_vars = ready_vars.union(map_reduce_outputs)
        while reduce_i < layer_count:
            op = step_layers[reduce_i][2]
            if op is None:
                reduce_i += 1
            elif len(set(op.inputs) & ready_vars) == len(op.inputs):
                step.reduce_ops.append(op)
                ready_vars = ready_vars.union(op.outputs)
                reduce_i += 1
            else:
                break
        steps.append(step)

    task = Task(
        name=constructor.name,
        dataset=dataset,
        type=constructor.type,
        steps=steps,
        inputs=inputs,
        outputs=list(constructor.outputs),
        strategy=constructor.strategy,
    )

    final_step = task.steps[-1]
    if len(final_step.map_reduce_ops) == 0 and len(final_step.reduce_ops) == 0:
        raise ValueError(
            "Should use a map reduce operator or a reduce operator as result"
        )
    if not all(var.location == DataLocation.SERVER for var in task.outputs):
        raise ValueError("All outputs should be on the server")

    return task
