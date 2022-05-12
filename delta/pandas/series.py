from __future__ import annotations

from typing import Any, Callable, List, Dict, Literal

from delta.core.task import (
    DataFormat,
    DataLocation,
    DataNode,
    GraphNode,
    InputGraphNode,
    MapOperator,
    MapReduceOperator,
    Operator,
    ReduceOperator,
)

import pandas

from .op_mixin import OpMixin, local_op, rlocal_op

SeriesAxisType = Literal[0]


class Series(InputGraphNode, OpMixin["Series"]):
    def __init__(
        self,
        name: str = "",
        location: DataLocation = DataLocation.CLIENT,
        src: Operator | None = None,
        dataset: str | None = None,
        default: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name, location, src, DataFormat.SERIES, dataset, default, **kwargs
        )

    def _dispatch_binary_op(self, other, name: str, op: Callable[..., Any], **kwargs):
        if not isinstance(other, GraphNode):
            other_node = InputGraphNode(default=other, location=DataLocation.SERVER)
        else:
            other_node = other

        if (
            self.location == DataLocation.CLIENT
            or other_node.location == DataLocation.CLIENT
        ):
            location = DataLocation.CLIENT
        else:
            location = DataLocation.SERVER

        output = Series(location=location)

        if location == DataLocation.SERVER:

            class _Reduce(ReduceOperator):
                def __init__(
                    self,
                    name: str,
                    inputs: List[DataNode],
                    outputs: List[DataNode],
                    op: Callable[..., Any],
                    **kwargs: Any,
                ) -> None:
                    super().__init__(name, inputs, outputs, **kwargs)
                    self.op = op

                def reduce(
                    self, left: pandas.Series, right: pandas.Series | float
                ) -> pandas.Series:
                    return self.op(left, right, **self.kwargs)

            _op = _Reduce(
                name=name, inputs=[self, other_node], outputs=[output], op=op, **kwargs
            )
            output.src = _op
        else:

            class _Map(MapOperator):
                def __init__(
                    self,
                    name: str,
                    inputs: List[DataNode],
                    outputs: List[DataNode],
                    op: Callable[..., Any],
                    **kwargs: Any,
                ) -> None:
                    super().__init__(name, inputs, outputs, **kwargs)
                    self.op = op

                def map(
                    self, left: pandas.Series, right: pandas.Series | float
                ) -> pandas.Series:
                    return self.op(left, right, **self.kwargs)

            _op = _Map(
                name=name, inputs=[self, other_node], outputs=[output], op=op, **kwargs
            )
            output.src = _op

        return output

    def _dispatch_map_reduce(
        self,
        name: str,
        map_reduce_op: MapReduceOperator,
        reduce_op: ReduceOperator | None = None,
        **kwargs,
    ):
        output = GraphNode(location=DataLocation.SERVER)

        if self.location == DataLocation.SERVER:
            if reduce_op is None:

                class _Reduce(ReduceOperator):
                    def reduce(self, data: pandas.Series) -> float:
                        return getattr(data, self.name)(**self.kwargs)

                reduce_op = _Reduce(
                    name=name, inputs=[self], outputs=[output], **kwargs
                )
            else:
                assert len(reduce_op.outputs) == 0
                reduce_op.outputs.append(output)

            output.src = reduce_op

        else:
            assert (
                len(map_reduce_op.reduce_outputs) == 0
                and len(map_reduce_op.outputs) == 0
            )
            map_reduce_op.reduce_outputs.append(output)
            map_reduce_op.outputs.append(output)
            output.src = map_reduce_op

        return output

    def all(
        self,
        axis: SeriesAxisType = 0,
        bool_only: bool = False,
        skipna: bool = True,
    ):
        class _MapReduce(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                axis: SeriesAxisType,
                bool_only: bool,
                skipna: bool,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    axis=axis,
                    bool_only=bool_only,
                    skipna=skipna,
                )
                self.axis: SeriesAxisType = axis
                self.bool_only = bool_only
                self.skipna = skipna

            def map(self, data: pandas.Series) -> Dict[str, int]:
                res = data.all(
                    axis=self.axis, bool_only=self.bool_only, skipna=self.skipna
                )
                return {"all": int(res)}

            def reduce(self, results: Dict[str, int], node_count: int) -> bool:
                agg_all = results.get("all")
                if agg_all is not None:
                    return agg_all == node_count
                else:
                    raise ValueError("aggregate result miss key all")

        map_reduce_op = _MapReduce(
            "all",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
            axis=axis,
            bool_only=bool_only,
            skipna=skipna,
        )

        return self._dispatch_map_reduce(
            name="all",
            map_reduce_op=map_reduce_op,
            axis=axis,
            bool_only=bool_only,
            skipna=skipna,
        )

    def any(
        self,
        axis: SeriesAxisType = 0,
        bool_only: bool = False,
        skipna: bool = True,
    ):
        class _MapReduce(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                axis: SeriesAxisType,
                bool_only: bool,
                skipna: bool,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    axis=axis,
                    bool_only=bool_only,
                    skipna=skipna,
                )
                self.axis: SeriesAxisType = axis
                self.bool_only = bool_only
                self.skipna = skipna

            def map(self, data: pandas.Series) -> Dict[str, int]:
                res = data.any(
                    axis=self.axis, bool_only=self.bool_only, skipna=self.skipna
                )
                return {"all": int(res)}

            def reduce(self, results: Dict[str, int], node_count: int) -> bool:
                agg_all = results.get("all")
                if agg_all is not None:
                    return agg_all > 0
                else:
                    raise ValueError("aggregate result miss key all")

        map_reduce_op = _MapReduce(
            "any",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
            axis=axis,
            bool_only=bool_only,
            skipna=skipna,
        )

        return self._dispatch_map_reduce(
            name="any",
            map_reduce_op=map_reduce_op,
            axis=axis,
            bool_only=bool_only,
            skipna=skipna,
        )

    def count(self):
        class _MapReduce(MapReduceOperator):
            def map(self, data: pandas.Series) -> Dict[str, int]:
                res = data.count()
                return {"count": res}

            def reduce(self, data: Dict[str, float], node_count: int) -> int:
                count = data.get("count")
                if count is not None:
                    return int(count)
                else:
                    raise ValueError("aggregate result miss key count")

        map_reduce_op = _MapReduce(
            name="count",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
        )

        return self._dispatch_map_reduce(
            name="count",
            map_reduce_op=map_reduce_op,
        )

    def sum(
        self,
        axis: SeriesAxisType = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
    ):
        class _MapReduce(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                axis: SeriesAxisType,
                skipna: bool,
                numeric_only: bool,
                min_count: int,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                )
                self.axis: SeriesAxisType = axis
                self.skipna = skipna
                self.numeric_only = numeric_only
                self.min_count = min_count

            def map(self, data: pandas.Series) -> Dict[str, float]:
                res = data.sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                    min_count=self.min_count,
                )
                return {"sum": res}

            def reduce(self, data: Dict[str, float], node_count: int) -> float:
                res = data.get("sum")
                if res is not None:
                    return res
                else:
                    raise ValueError("aggregate result miss key sum")

        map_reduce_op = _MapReduce(
            name="sum",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
        )

        return self._dispatch_map_reduce(
            name="sum",
            map_reduce_op=map_reduce_op,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
        )

    def mean(
        self,
        axis: SeriesAxisType = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ):
        class _MapReduce(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                axis: SeriesAxisType,
                skipna: bool,
                numeric_only: bool,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                )
                self.axis: SeriesAxisType = axis
                self.skipna = skipna
                self.numeric_only = numeric_only

            def map(self, data: pandas.Series) -> Dict[str, float]:
                sum = data.sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                )
                count = data.count()
                if not self.skipna:
                    count += data.isna().sum(axis=self.axis)
                return {"sum": sum, "count": count}

            def reduce(self, data: Dict[str, float], node_count: int) -> float:
                sum = data.get("sum")
                if sum is None:
                    raise ValueError("aggregate result miss key sum")
                count = data.get("count")
                if count is None:
                    raise ValueError("aggregate result miss key count")

                return sum / count

        map_reduce_op = _MapReduce(
            name="mean",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
        )

        return self._dispatch_map_reduce(
            name="mean",
            map_reduce_op=map_reduce_op,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
        )

    def std(
        self,
        axis: SeriesAxisType = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        class _MapReduce(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                axis: SeriesAxisType,
                skipna: bool,
                ddof: int,
                numeric_only: bool,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    axis=axis,
                    skipna=skipna,
                    ddof=ddof,
                    numeric_only=numeric_only,
                )
                self.axis: SeriesAxisType = axis
                self.skipna = skipna
                self.ddof = ddof
                self.numeric_only = numeric_only

            def map(self, data: pandas.Series) -> Dict[str, float]:
                sum1 = data.sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                )
                sum2 = data.pow(2).sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                )
                count = data.count()
                if not self.skipna:
                    count += data.isna().sum(axis=self.axis)
                return {"sum1": sum1, "sum2": sum2, "count": count}

            def reduce(self, data: Dict[str, float], node_count: int) -> float:
                sum1 = data.get("sum1")
                if sum1 is None:
                    raise ValueError("aggregate result miss key sum1")
                sum2 = data.get("sum2")
                if sum2 is None:
                    raise ValueError("aggregate result miss key sum2")
                count = data.get("count")
                if count is None:
                    raise ValueError("aggregate result miss key count")

                res = (sum2 / count - (sum1 / count) ** 2) * count / (count - self.ddof)
                return res ** (1 / 2)

        map_reduce_op = _MapReduce(
            name="std",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

        return self._dispatch_map_reduce(
            name="std",
            map_reduce_op=map_reduce_op,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    def var(
        self,
        axis: SeriesAxisType = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        class _MapReduce(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                axis: SeriesAxisType,
                skipna: bool,
                ddof: int,
                numeric_only: bool,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    axis=axis,
                    skipna=skipna,
                    ddof=ddof,
                    numeric_only=numeric_only,
                )
                self.axis: SeriesAxisType = axis
                self.skipna = skipna
                self.ddof = ddof
                self.numeric_only = numeric_only

            def map(self, data: pandas.Series) -> Dict[str, float]:
                sum1 = data.sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                )
                sum2 = data.pow(2).sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                )
                count = data.count()
                if not self.skipna:
                    count += data.isna().sum(axis=self.axis)
                return {"sum1": sum1, "sum2": sum2, "count": count}

            def reduce(self, data: Dict[str, float], node_count: int) -> float:
                sum1 = data.get("sum1")
                if sum1 is None:
                    raise ValueError("aggregate result miss key sum1")
                sum2 = data.get("sum2")
                if sum2 is None:
                    raise ValueError("aggregate result miss key sum2")
                count = data.get("count")
                if count is None:
                    raise ValueError("aggregate result miss key count")

                res = (sum2 / count - (sum1 / count) ** 2) * count / (count - self.ddof)
                return res

        map_reduce_op = _MapReduce(
            name="var",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

        return self._dispatch_map_reduce(
            name="var",
            map_reduce_op=map_reduce_op,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    def sem(
        self,
        axis: SeriesAxisType = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        class _MapReduce(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                axis: SeriesAxisType,
                skipna: bool,
                ddof: int,
                numeric_only: bool,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    axis=axis,
                    skipna=skipna,
                    ddof=ddof,
                    numeric_only=numeric_only,
                )
                self.axis: SeriesAxisType = axis
                self.skipna = skipna
                self.ddof = ddof
                self.numeric_only = numeric_only

            def map(self, data: pandas.Series) -> Dict[str, float]:
                sum1 = data.sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                )
                sum2 = data.pow(2).sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                )
                count = data.count()
                if not self.skipna:
                    count += data.isna().sum(axis=self.axis)
                return {"sum1": sum1, "sum2": sum2, "count": count}

            def reduce(self, data: Dict[str, float], node_count: int) -> float:
                sum1 = data.get("sum1")
                if sum1 is None:
                    raise ValueError("aggregate result miss key sum1")
                sum2 = data.get("sum2")
                if sum2 is None:
                    raise ValueError("aggregate result miss key sum2")
                count = data.get("count")
                if count is None:
                    raise ValueError("aggregate result miss key count")

                res = ((sum2 / count - (sum1 / count) ** 2) / (count - self.ddof)) ** (
                    1 / 2
                )
                return res

        map_reduce_op = _MapReduce(
            name="sem",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

        return self._dispatch_map_reduce(
            name="sem",
            map_reduce_op=map_reduce_op,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    def _arith_method(self, other, name: str, op: Callable[[Any, Any], Any]):
        return self._dispatch_binary_op(other, name, op)

    def add(self, other: "Series", fill_value: float | None = None, axis=0) -> "Series":
        return self._dispatch_binary_op(
            other, "add", local_op("add"), axis=axis, fill_value=fill_value
        )

    def sub(self, other: "Series", fill_value: float | None = None, axis=0) -> "Series":
        return self._dispatch_binary_op(
            other, "sub", local_op("sub"), axis=axis, fill_value=fill_value
        )

    def mul(self, other: "Series", fill_value: float | None = None, axis=0) -> "Series":
        return self._dispatch_binary_op(
            other, "mul", local_op("mul"), axis=axis, fill_value=fill_value
        )

    def div(self, other: "Series", fill_value: float | None = None, axis=0) -> "Series":
        return self._dispatch_binary_op(
            other, "div", local_op("div"), axis=axis, fill_value=fill_value
        )

    def truediv(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "truediv", local_op("truediv"), axis=axis, fill_value=fill_value
        )

    def floordiv(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "floordiv", local_op("floordiv"), axis=axis, fill_value=fill_value
        )

    def mod(self, other: "Series", fill_value: float | None = None, axis=0) -> "Series":
        return self._dispatch_binary_op(
            other, "mod", local_op("mod"), axis=axis, fill_value=fill_value
        )

    def pow(self, other: "Series", fill_value: float | None = None, axis=0) -> "Series":
        return self._dispatch_binary_op(
            other, "pow", local_op("pow"), axis=axis, fill_value=fill_value
        )

    def radd(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "radd", rlocal_op("radd"), axis=axis, fill_value=fill_value
        )

    def rsub(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "rsub", rlocal_op("rsub"), axis=axis, fill_value=fill_value
        )

    def rmul(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "rmul", rlocal_op("rmul"), axis=axis, fill_value=fill_value
        )

    def rdiv(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "rdiv", rlocal_op("rdiv"), axis=axis, fill_value=fill_value
        )

    def rtruediv(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "rtruediv", rlocal_op("rtruediv"), axis=axis, fill_value=fill_value
        )

    def rfloordiv(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other,
            "rfloordiv",
            rlocal_op("rfloordiv"),
            axis=axis,
            fill_value=fill_value,
        )

    def rmod(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "rmod", rlocal_op("rmod"), axis=axis, fill_value=fill_value
        )

    def rpow(
        self, other: "Series", fill_value: float | None = None, axis=0
    ) -> "Series":
        return self._dispatch_binary_op(
            other, "rpow", rlocal_op("rpow"), axis=axis, fill_value=fill_value
        )
