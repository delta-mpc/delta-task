from __future__ import annotations

from typing import Callable, List, Dict, Tuple, Literal, Optional

import numpy as np
import numpy.typing as npt

from delta.core.task import (
    GraphNode,
    DataNode,
    MapReduceOperator,
    DataLocation,
    EarlyStop,
    InputGraphNode,
)

FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]

OrderACF = Optional[Literal["A", "C", "F"]]

ModelFunc = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]


def norm(v: FloatArray, ord: float = 2):
    if ord == np.inf:
        return np.max(np.abs(v))
    elif ord == -np.inf:
        return np.min(np.abs(v))
    else:
        return np.sum(np.abs(v) ** ord, axis=0) ** (1 / ord)


def newton_step(
    f: ModelFunc,
    score: ModelFunc,
    hessian: ModelFunc,
    x_node: GraphNode,
    y_node: GraphNode,
    params_node: GraphNode,
    iteration_node: GraphNode,
    order: OrderACF = None,
    ord: float = np.inf,
    tol: float = 1e-8,
    ridge_factor: float = 1e-10,
):
    new_f_val_node = GraphNode(name="f_val", location=DataLocation.SERVER)
    new_params_node = GraphNode(name="params", location=DataLocation.SERVER)
    new_iteration_node = GraphNode(name="iteration", location=DataLocation.SERVER)

    class NewtonStep(MapReduceOperator):
        def __init__(
            self,
            name: str,
            map_inputs: List[DataNode],
            reduce_inputs: List[DataNode],
            map_outputs: List[DataNode],
            reduce_outputs: List[DataNode],
            f: ModelFunc,
            score: ModelFunc,
            hessian: ModelFunc,
            order: OrderACF = None,
            ord: float = np.inf,
            tol: float = 1e-8,
            ridge_factor: float = 1e-10,
        ) -> None:
            super().__init__(
                name,
                map_inputs,
                reduce_inputs,
                map_outputs,
                reduce_outputs,
                f=f,
                score=score,
                hessian=hessian,
                order=order,
                ord=ord,
                tol=tol,
                ridge_factor=ridge_factor,
            )
            self.f = f
            self.score = score
            self.hessian = hessian
            self.order: OrderACF = order
            self.ord = ord
            self.tol = tol
            self.ridge_factor = ridge_factor

        def map(
            self,
            x: FloatArray,
            y: FloatArray,
            params: FloatArray,
        ) -> Dict[str, FloatArray]:
            obs = x.shape[0]
            F = self.f(params, y, x)
            G = self.score(params, y, x)
            H = self.hessian(params, y, x)
            return {
                "f": F,
                "score": G,
                "hessian": H,
                "n": np.asarray(obs),
            }

        def reduce(
            self,
            data: Dict[str, FloatArray],
            node_count: int,
            params: FloatArray,
            iteration: int,
        ):
            n = data["n"]
            f = data["f"] / n
            score = data["score"] / n
            hessian = data["hessian"] / n

            if norm(score, ord=self.ord) < self.tol:
                raise EarlyStop

            if not np.all(self.ridge_factor == 0):
                hessian[np.diag_indices_from(hessian)] += ridge_factor

            d: FloatArray = np.linalg.solve(hessian, score)
            d = d.reshape(params.shape, order=self.order)
            params = params - d
            return params, f, iteration + 1

    step_op = NewtonStep(
        name="fit_newton_step",
        map_inputs=[x_node, y_node, params_node],
        reduce_inputs=[params_node, iteration_node],
        map_outputs=[],
        reduce_outputs=[new_params_node, new_f_val_node, new_iteration_node],
        f=f,
        score=score,
        hessian=hessian,
        ridge_factor=ridge_factor,
        order=order,
        ord=ord,
        tol=tol,
    )
    new_params_node.src = step_op
    new_f_val_node.src = step_op
    new_iteration_node.src = step_op
    return new_params_node, new_f_val_node, new_iteration_node


def fit_newton(
    f: ModelFunc,
    score: ModelFunc,
    hessian: ModelFunc,
    x_node: GraphNode,
    y_node: GraphNode,
    start_params_node: GraphNode,
    maxiter: int = 100,
    order: OrderACF = None,
    ord: float = np.inf,
    tol: float = 1e-8,
    ridge_factor: float = 1e-10,
) -> Tuple[GraphNode, GraphNode, GraphNode]:
    params_node = start_params_node
    f_val_node = InputGraphNode(
        name="f_val", location=DataLocation.SERVER, default=np.inf
    )
    iteration_node = InputGraphNode(
        name="iteration", location=DataLocation.SERVER, default=0
    )
    for _ in range(maxiter):
        params_node, f_val_node, iteration_node = newton_step(
            f,
            score,
            hessian,
            x_node,
            y_node,
            params_node,
            iteration_node,
            order,
            ord,
            tol,
            ridge_factor,
        )
    return params_node, f_val_node, iteration_node


def fit(
    f: ModelFunc,
    score: ModelFunc,
    x_node: GraphNode,
    y_node: GraphNode,
    start_params_node: GraphNode,
    method: str = "newton",
    hessian: ModelFunc | None = None,
    maxiter: int = 100,
    order: OrderACF = None,
    ord: float = np.inf,
    tol: float = 1e-8,
    ridge_factor: float = 1e-10,
) -> Tuple[GraphNode, GraphNode, GraphNode]:
    if method == "newton":
        assert hessian is not None
        return fit_newton(
            f,
            score,
            hessian,
            x_node,
            y_node,
            start_params_node,
            maxiter,
            order,
            ord,
            tol,
            ridge_factor,
        )
    else:
        raise ValueError(f"unknown method {method}")
