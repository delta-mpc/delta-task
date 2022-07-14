from abc import abstractmethod
from typing import Any, Tuple, Dict, List, Callable

import numpy as np
import numpy.typing as npt
from delta.task import HorizontalTask
from delta.core.strategy import AnalyticsStrategy, CURVE_TYPE
from delta.core.task import (
    InputGraphNode,
    GraphNode,
    DataLocation,
    DataNode,
    MapOperator,
    MapReduceOperator,
)

from .optimizer import fit

__all__ = ["MNLogitTask"]

FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]


def softmax(x: FloatArray):
    x_ = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x_) / np.sum(np.exp(x_), axis=1, keepdims=True)


def logsoftmax(x: FloatArray):
    x_ = x - np.max(x, axis=1, keepdims=True)
    return x_ - np.log(np.sum(np.exp(x_), axis=1, keepdims=True))


def convert_params(params: FloatArray, y: IntArray, x: FloatArray) -> FloatArray:
    if params.shape[0] != x.shape[1]:
        params = params.reshape(x.shape[1], -1, order="F")
    assert params.shape[1] == y.shape[1] - 1
    return params


def loglike(params: FloatArray, y: IntArray, x: FloatArray):
    params = convert_params(params, y, x)
    p = np.column_stack((np.zeros(x.shape[0]), np.dot(x, params)))
    return np.sum(y * logsoftmax(p))


def loglikeobs(params: FloatArray, y: IntArray, x: FloatArray):
    params = convert_params(params, y, x)
    p = np.column_stack((np.zeros(x.shape[0]), np.dot(x, params)))
    return y * logsoftmax(p)


def score(params: FloatArray, y: IntArray, x: FloatArray):
    params = convert_params(params, y, x)
    p = np.column_stack((np.zeros(x.shape[0]), np.dot(x, params)))
    return np.dot(x.T, y - softmax(p))[:, 1:].ravel(order="F")


def score_obs(params: FloatArray, y: IntArray, x: FloatArray):
    params = convert_params(params, y, x)
    p = np.column_stack((np.zeros(x.shape[0]), np.dot(x, params)))
    d = (y - softmax(p))[:, 1:]  # shape [m, j - 1]
    return (d[:, :, np.newaxis] * x[:, np.newaxis, :]).reshape(x.shape[0], -1)


def hessian(params: FloatArray, y: IntArray, x: FloatArray):
    params = convert_params(params, y, x)
    p = np.column_stack((np.zeros(x.shape[0]), np.dot(x, params)))
    a = softmax(p)

    J = y.shape[1]
    K = x.shape[1]
    C = (J - 1) * K
    partials = []
    for i in range(J - 1):
        for j in range(J - 1):
            if i == j:
                partials.append(
                    np.dot(
                        ((a[:, i + 1] * (a[:, j + 1] - 1))[:, np.newaxis] * x).T,
                        x,
                    )
                )
            else:
                partials.append(
                    np.dot(
                        ((a[:, i + 1] * a[:, j + 1])[:, np.newaxis] * x).T,
                        x,
                    )
                )

    res = np.array(partials)
    res = np.transpose(res.reshape(J - 1, J - 1, K, K), (0, 2, 1, 3)).reshape(C, C)
    return res


class MNLogitTask(HorizontalTask):
    def __init__(
        self,
        name: str,
        min_clients: int = 2,
        max_clients: int = 2,
        wait_timeout: float = 60,
        connection_timeout: float = 60,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
        max_n_classes: int = 100,
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
        self.max_n_classes = max_n_classes

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
            order="F",
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
                y = np.asarray(y, dtype=np.int32)
                assert x.shape[0] == y.shape[0], "x and y should have same items"
                assert y.ndim == 1, "y should be an 1-D array"
                return x, y

        x_node = GraphNode(
            name="x",
            location=DataLocation.CLIENT,
        )
        origin_y_node = GraphNode(
            name="origin_y",
            location=DataLocation.CLIENT,
        )

        preprocess_op = Preprocess(
            name="preprocess",
            inputs=list(inputs.values()),
            outputs=[x_node, origin_y_node],
            preprocess=self.preprocess,
            names=list(inputs.keys()),
        )
        x_node.src = preprocess_op
        origin_y_node.src = preprocess_op

        n_classes_node = GraphNode(
            name="n_classes",
            location=DataLocation.SERVER,
        )

        class NClassesOp(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                max_n_classes: int
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    max_n_classes=max_n_classes,
                )
                self.max_n_classes = max_n_classes

            def map(self, y: IntArray) -> Dict[str, IntArray]:
                bins = np.zeros((self.max_n_classes,), dtype=np.int8)
                for label in y:
                    bins[label] = 1
                return {"bins": bins}

            def reduce(self, data: Dict[str, IntArray], node_count: int):
                bins = data["bins"]
                assert bins[0] > 0, "y should start from 0"
                max_label = 0
                for label in range(len(bins) - 1, -1, -1):
                    if bins[label] > 0:
                        max_label = label
                        break
                return max_label + 1
        
        n_classes_op = NClassesOp(
            name="n_classes_op",
            map_inputs=[origin_y_node],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[n_classes_node],
            max_n_classes=self.max_n_classes
        )
        n_classes_node.src = n_classes_op

        y_node = GraphNode(name="y", location=DataLocation.CLIENT)

        class OnehotOp(MapOperator):
            def map(self, y: IntArray, n_classes: int) -> IntArray:
                m = y.shape[0]
                res = np.zeros((m, n_classes), dtype=np.int8)
                res[np.arange(m), y] = 1
                return res

        one_hot_op = OnehotOp(
            name="one_hot",
            inputs=[origin_y_node, n_classes_node],
            outputs=[y_node]
        )
        y_node.src = one_hot_op

        options = self.options()
        method = options.pop("method", "newton")
        maxiter = options.pop("maxiter", 35)
        start_params = options.pop("start_params", None)

        if start_params is None:
            params_node = GraphNode(name="params", location=DataLocation.SERVER)

            class ParamsInitOp(MapReduceOperator):
                def map(self, x: FloatArray, y: IntArray):
                    params = np.zeros((x.shape[1], y.shape[1] - 1), dtype=np.float64)
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

            class ParamsConvertOp(MapReduceOperator):
                def map(self, params: FloatArray, x: FloatArray, y: IntArray):
                    params = convert_params(params, y, x)
                    return {"params": params}

                def reduce(self, data, node_count: int):
                    params = data["params"]
                    return params

            params_check_op = ParamsConvertOp(
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
