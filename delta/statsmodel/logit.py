from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from delta.core.strategy import CURVE_TYPE, AnalyticsStrategy
from delta.core.task import (
    DataLocation,
    DataNode,
    GraphNode,
    InputGraphNode,
    MapOperator,
    MapReduceOperator,
    TaskType,
)
from delta.task import HorizontalTask

from .optimizer import fit

__all__ = ["LogitTask"]

FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]


def sigmoid(x: FloatArray) -> FloatArray:
    out = np.zeros_like(x)

    con1 = np.abs(x) >= 5
    out[con1] = 1
    con2 = np.logical_and(np.abs(x) < 5, np.abs(x) >= 2.375)
    out[con2] = 0.03125 * np.abs(x[con2]) + 0.84375
    con3 = np.logical_and(np.abs(x) < 2.375, np.abs(x) >= 1)
    out[con3] = 0.125 * np.abs(x[con3]) + 0.625
    con4 = np.logical_and(np.abs(x) < 1, np.abs(x) >= 0)
    out[con4] = 0.25 * np.abs(x[con4]) + 0.5
    con5 = x < 0
    out[con5] = 1 - out[con5]
    return out


def logsigmoid(x: FloatArray) -> FloatArray:
    return x - np.log(1 + np.exp(x))


def check_params(params: FloatArray, x: FloatArray):
    assert params.ndim == 1
    assert params.size == x.shape[1]


def loglike(params: FloatArray, y: IntArray, x: FloatArray):
    p = (2 * y - 1) * np.dot(x, params)
    return np.sum(logsigmoid(p))


def loglikeobs(params: FloatArray, y: IntArray, x: FloatArray):
    p = (2 * y - 1) * np.dot(x, params)
    return logsigmoid(p)


def score(params: FloatArray, y: IntArray, x: FloatArray):
    p = np.dot(x, params)
    return np.dot(x.T, y - sigmoid(p))


def score_obs(params: FloatArray, y: IntArray, x: FloatArray):
    p = np.dot(x, params)
    return (y - sigmoid(p)) * x


def hessian(params: FloatArray, y: IntArray, x: FloatArray):
    p = np.dot(x, params)
    l = sigmoid(p)
    ll = l * (1 - l)
    return -np.dot(x.T * ll, x)


class LogitTask(HorizontalTask):
    TYPE: TaskType = TaskType.HLR

    def __init__(
        self,
        name: str,
        min_clients: int = 2,
        max_clients: int = 2,
        wait_timeout: float = 60,
        connection_timeout: float = 60,
        verify_timeout: float = 300,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
        enable_verify: bool = True,
    ) -> None:
        strategy = AnalyticsStrategy(
            min_clients=min_clients,
            max_clients=max_clients,
            wait_timeout=wait_timeout,
            connection_timeout=connection_timeout,
            verify_timeout=verify_timeout,
            precision=precision,
            curve=curve,
        )
        super().__init__(name, strategy, enable_verify)

    @abstractmethod
    def preprocess(self, **inputs: Any) -> Tuple[Any, Any]:
        ...

    @abstractmethod
    def dataset(self) -> Dict[str, InputGraphNode]:
        ...

    def options(self) -> Dict[str, Any]:
        return {
            "maxiter": 35,
            "method": "newton",
            "ord": np.inf,
            "tol": 1e-8,
            "ridge_factor": 1e-10,
        }

    def _fit(
        self,
        x_node: GraphNode,
        y_node: GraphNode,
        params_node: GraphNode,
        method: str = "newton",
        maxiter: int = 35,
        **kwargs,
    ):
        def f(params, y, x):
            return -loglike(params, y, x)

        def g(params, y, x):
            return -score(params, y, x)

        def h(params, y, x):
            return -hessian(params, y, x)

        opt_params, f_opt, iteration = fit(
            f,
            g,
            x_node,
            y_node,
            params_node,
            method,
            hessian=h,
            maxiter=maxiter,
            **kwargs,
        )
        return opt_params, f_opt, iteration

    def _build_graph(self) -> Tuple[List[InputGraphNode], List[GraphNode]]:
        inputs = self.dataset()
        input_nodes: List[InputGraphNode] = list(inputs.values())
        for name, node in inputs.items():
            node.name = name

        class Preprocess(MapOperator):
            def __init__(
                self,
                name: str,
                inputs: List[DataNode],
                outputs: List[DataNode],
                preprocess: Callable,
                names: List[str],
            ) -> None:
                super().__init__(
                    name, inputs, outputs, preprocess=preprocess, names=names
                )
                self.preprocess = preprocess
                self.names = names

            def map(self, *args) -> Tuple[FloatArray, IntArray]:
                kwargs = dict(zip(self.names, args))
                x, y = self.preprocess(**kwargs)
                x = np.asarray(x, dtype=np.float64)
                assert x.ndim == 2, "x can only be in dim 2"
                y = np.asarray(y, dtype=np.int8).squeeze()
                assert x.shape[0] == y.shape[0], "x and y should have same items"
                return x, y

        x_node = GraphNode(
            name="x",
            location=DataLocation.CLIENT,
        )
        y_node = GraphNode(
            name="y",
            location=DataLocation.CLIENT,
        )

        preprocess_op = Preprocess(
            name="preprocess",
            inputs=list(inputs.values()),
            outputs=[x_node, y_node],
            preprocess=self.preprocess,
            names=list(inputs.keys()),
        )
        x_node.src = preprocess_op
        y_node.src = preprocess_op

        options = self.options()
        method = options.pop("method", "newton")
        maxiter = options.pop("maxiter", 35)
        start_params = options.pop("start_params", None)

        if start_params is None:
            params_node = GraphNode(name="params", location=DataLocation.SERVER)

            class ParamsInitOp(MapReduceOperator):
                def map(self, x: FloatArray, y: IntArray):
                    params = np.zeros((x.shape[1],), dtype=np.float64)
                    return {"params": params}

                def reduce(self, data, node_count: int):
                    params = data["params"]
                    return params

            params_init_op = ParamsInitOp(
                name="params_init",
                map_inputs=[x_node, y_node],
                reduce_inputs=[],
                map_outputs=[],
                reduce_outputs=[params_node],
            )
            params_node.src = params_init_op
        else:
            origin_params_node = InputGraphNode(
                name="params", location=DataLocation.SERVER, default=start_params
            )
            params_node = GraphNode(name="params", location=DataLocation.SERVER)

            class ParamsCheckOp(MapReduceOperator):
                def map(self, params: FloatArray, x: FloatArray, y: IntArray):
                    assert check_params(params, x)
                    return {"params": params}

                def reduce(self, data, node_count: int):
                    params = data["params"]
                    return params

            params_check_op = ParamsCheckOp(
                name="params_check",
                map_inputs=[origin_params_node, x_node, y_node],
                reduce_inputs=[],
                map_outputs=[],
                reduce_outputs=[params_node],
            )
            params_node.src = params_check_op

            input_nodes.append(origin_params_node)

        outputs = self._fit(x_node, y_node, params_node, method, maxiter, **options)

        return input_nodes, list(outputs)
