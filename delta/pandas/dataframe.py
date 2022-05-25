from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal

import pandas

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

from .series import Series
from .op_mixin import OpMixin, local_op, rlocal_op

AxisType = Literal[0, 1]


class DataFrame(InputGraphNode, OpMixin["DataFrame"]):
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
            name, location, src, DataFormat.DATAFRAME, dataset, default, **kwargs
        )

    def _dispatch_map_reduce(
        self,
        name: str,
        axis: AxisType,
        map_reduce_op: MapReduceOperator,
        map_op: MapOperator | None = None,
        reduce_op: ReduceOperator | None = None,
        reduce_axis: AxisType = 0,
        **kwargs,
    ):
        output = Series(location=DataLocation.CLIENT)

        if self.location == DataLocation.SERVER:
            if reduce_op is None:

                class _Reduce(ReduceOperator):
                    def reduce(self, data: pandas.DataFrame) -> pandas.Series:
                        return getattr(data, self.name)(**self.kwargs)

                reduce_op = _Reduce(
                    name=name, inputs=[self], outputs=[output], **kwargs
                )
            else:
                assert len(reduce_op.outputs) == 0
                reduce_op.outputs.append(output)
            output.src = reduce_op
            output.location = DataLocation.SERVER

        elif axis == reduce_axis:
            assert (
                len(map_reduce_op.reduce_outputs) == 0
                and len(map_reduce_op.outputs) == 0
            )
            map_reduce_op.reduce_outputs.append(output)
            map_reduce_op.outputs.append(output)
            output.src = map_reduce_op
            output.location = DataLocation.SERVER

        else:
            if map_op is None:

                class _Map(MapOperator):
                    def map(self, data: pandas.DataFrame) -> pandas.Series:
                        return getattr(data, self.name)(**self.kwargs)

                map_op = _Map(name=name, inputs=[self], outputs=[output], **kwargs)
            else:
                assert len(map_op.outputs) == 0
                map_op.outputs.append(output)
            output.src = map_op

        return output

    def _dispatch_binary_op(
        self,
        other: "DataFrame" | "Series" | List[float] | float,
        op_name: str,
        op: Callable[..., Any],
        **kwargs: Any,
    ) -> "DataFrame":
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

        output = DataFrame(location=location)

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
                    self,
                    left: pandas.DataFrame,
                    right: pandas.DataFrame | pandas.Series | List[float] | float,
                ):
                    return self.op(left, right, **self.kwargs)

            _op = _Reduce(
                op_name, inputs=[self, other_node], outputs=[output], op=op, **kwargs
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
                    self,
                    left: pandas.DataFrame,
                    right: pandas.DataFrame | pandas.Series | List[float] | float,
                ):
                    return self.op(left, right, **self.kwargs)

            _op = _Map(
                op_name, inputs=[self, other_node], outputs=[output], op=op, **kwargs
            )
            output.src = _op

        return output

    def all(
        self,
        axis: AxisType = 0,
        bool_only: bool = False,
        skipna: bool = True,
    ) -> "Series":
        class _MapReduce(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                axis: AxisType = 0,
                bool_only: bool = False,
                skipna: bool = True,
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
                self.axis: AxisType = axis
                self.bool_only = bool_only
                self.skipna = skipna

            def map(self, data: pandas.DataFrame) -> Dict[str, pandas.Series]:
                res = data.all(
                    axis=self.axis, bool_only=self.bool_only, skipna=self.skipna
                )
                return {"all": res}

            def reduce(
                self, results: Dict[str, pandas.Series], node_count: int
            ) -> pandas.Series:
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
            axis=axis,
            map_reduce_op=map_reduce_op,
            reduce_axis=0,
            bool_only=bool_only,
            skipna=skipna,
        )

    def any(
        self,
        axis: AxisType = 0,
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
                axis: AxisType = 0,
                bool_only: bool = False,
                skipna: bool = True,
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
                self.axis: AxisType = axis
                self.bool_only = bool_only
                self.skipna = skipna

            def map(self, data: pandas.DataFrame) -> Dict[str, pandas.Series]:
                res = data.any(
                    axis=self.axis, bool_only=self.bool_only, skipna=self.skipna
                )
                return {"all": res}

            def reduce(
                self, results: Dict[str, pandas.Series], node_count: int
            ) -> pandas.Series:
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
            axis=axis,
            map_reduce_op=map_reduce_op,
            reduce_axis=0,
            bool_only=bool_only,
            skipna=skipna,
        )

    def count(
        self,
        axis: AxisType = 0,
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
                axis: AxisType,
                numeric_only: bool,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    axis=axis,
                    numeric_only=numeric_only,
                )
                self.axis: AxisType = axis
                self.numeric_only = numeric_only

            def map(self, data: pandas.DataFrame) -> Dict[str, pandas.Series]:
                res = data.count(axis=self.axis, numeric_only=self.numeric_only)
                return {"count": res}

            def reduce(
                self, data: Dict[str, pandas.Series], node_count: int
            ) -> pandas.Series:
                count = data.get("count")
                if count is not None:
                    return count
                else:
                    raise ValueError("aggregate result miss key count")

        map_reduce_op = _MapReduce(
            name="count",
            map_inputs=[self],
            reduce_inputs=[],
            map_outputs=[],
            reduce_outputs=[],
            axis=axis,
            numeric_only=numeric_only,
        )

        return self._dispatch_map_reduce(
            name="count",
            axis=axis,
            map_reduce_op=map_reduce_op,
            reduce_axis=0,
            numeric_only=numeric_only,
        )

    def sum(
        self,
        axis: AxisType = 0,
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
                axis: AxisType,
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
                self.axis: AxisType = axis
                self.skipna = skipna
                self.numeric_only = numeric_only
                self.min_count = min_count

            def map(self, data: pandas.DataFrame) -> Dict[str, pandas.Series]:
                res = data.sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                    min_count=self.min_count,
                )
                return {"sum": res}

            def reduce(
                self, data: Dict[str, pandas.Series], node_count: int
            ) -> pandas.Series:
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
            axis=axis,
            map_reduce_op=map_reduce_op,
            reduce_axis=0,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
        )

    def mean(
        self,
        axis: AxisType = 0,
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
                axis: AxisType,
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
                self.axis: AxisType = axis
                self.skipna = skipna
                self.numeric_only = numeric_only

            def map(self, data: pandas.DataFrame) -> Dict[str, pandas.Series]:
                sum = data.sum(
                    axis=self.axis,
                    skipna=self.skipna,
                    numeric_only=self.numeric_only,
                )
                count = data.count(axis=self.axis, numeric_only=self.numeric_only)
                if not self.skipna:
                    count += data.isna().sum(axis=self.axis)
                return {"sum": sum, "count": count}

            def reduce(
                self, data: Dict[str, pandas.Series], node_count: int
            ) -> pandas.Series:
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
            axis=axis,
            map_reduce_op=map_reduce_op,
            reduce_axis=0,
            skipna=skipna,
            numeric_only=numeric_only,
        )

    def std(
        self,
        axis: AxisType = 0,
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
                axis: AxisType,
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
                self.axis: AxisType = axis
                self.skipna = skipna
                self.ddof = ddof
                self.numeric_only = numeric_only

            def map(self, data: pandas.DataFrame) -> Dict[str, pandas.Series]:
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
                count = data.count(
                    axis=self.axis,
                    numeric_only=self.numeric_only,
                )
                if not self.skipna:
                    count += data.isna().sum(axis=self.axis)
                return {"sum1": sum1, "sum2": sum2, "count": count}

            def reduce(
                self, data: Dict[str, pandas.Series], node_count: int
            ) -> pandas.Series:
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
            axis=axis,
            map_reduce_op=map_reduce_op,
            reduce_axis=0,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    def var(
        self,
        axis: AxisType = 0,
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
                axis: AxisType,
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
                self.axis: AxisType = axis
                self.skipna = skipna
                self.ddof = ddof
                self.numeric_only = numeric_only

            def map(self, data: pandas.DataFrame) -> Dict[str, pandas.Series]:
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
                count = data.count(
                    axis=self.axis,
                    numeric_only=self.numeric_only,
                )
                if not self.skipna:
                    count += data.isna().sum(axis=self.axis)
                return {"sum1": sum1, "sum2": sum2, "count": count}

            def reduce(
                self, data: Dict[str, pandas.Series], node_count: int
            ) -> pandas.Series:
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
            axis=axis,
            map_reduce_op=map_reduce_op,
            reduce_axis=0,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    def sem(
        self,
        axis: AxisType = 0,
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
                axis: AxisType,
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
                    numeric_only=numeric_only
                )
                self.axis: AxisType = axis
                self.skipna = skipna
                self.ddof = ddof
                self.numeric_only = numeric_only

            def map(self, data: pandas.DataFrame) -> Dict[str, pandas.Series]:
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
                count = data.count(
                    axis=self.axis,
                    numeric_only=self.numeric_only,
                )
                if not self.skipna:
                    count += data.isna().sum(axis=self.axis)
                return {"sum1": sum1, "sum2": sum2, "count": count}

            def reduce(self, data: Dict[str, pandas.Series], node_count: int) -> pandas.Series:
                sum1 = data.get("sum1")
                if sum1 is None:
                    raise ValueError("aggregate result miss key sum1")
                sum2 = data.get("sum2")
                if sum2 is None:
                    raise ValueError("aggregate result miss key sum2")
                count = data.get("count")
                if count is None:
                    raise ValueError("aggregate result miss key count")

                res = ((sum2 / count - (sum1 / count) ** 2) / (count - self.ddof)) ** (1 / 2)
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
            axis=axis,
            map_reduce_op=map_reduce_op,
            reduce_axis=0,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    def _arith_method(
        self, other, name: str, op: Callable[[Any, Any], Any]
    ) -> "DataFrame":
        return self._dispatch_binary_op(other, name, op)

    def add(
        self, other: "DataFrame", axis: AxisType = 1, fill_value: float | None = None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "add", local_op("add"), axis=axis, fill_value=fill_value
        )

    def sub(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "sub", local_op("sub"), axis=axis, fill_value=fill_value
        )

    def mul(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "mul", local_op("mul"), axis=axis, fill_value=fill_value
        )

    def div(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "div", local_op("div"), axis=axis, fill_value=fill_value
        )

    def truediv(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "truediv", local_op("truediv"), axis=axis, fill_value=fill_value
        )

    def floordiv(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "floordiv", local_op("floordiv"), axis=axis, fill_value=fill_value
        )

    def mod(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "mod", local_op("mod"), axis=axis, fill_value=fill_value
        )

    def pow(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "pow", local_op("pow"), axis=axis, fill_value=fill_value
        )

    def radd(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "radd", rlocal_op("radd"), axis=axis, fill_value=fill_value
        )

    def rsub(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "rsub", rlocal_op("rsub"), axis=axis, fill_value=fill_value
        )

    def rmul(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "rmul", rlocal_op("rmul"), axis=axis, fill_value=fill_value
        )

    def rdiv(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "rdiv", rlocal_op("rdiv"), axis=axis, fill_value=fill_value
        )

    def rtruediv(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "rtruediv", rlocal_op("rtruediv"), axis=axis, fill_value=fill_value
        )

    def rfloordiv(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other,
            "rfloordiv",
            rlocal_op("rfloordiv"),
            axis=axis,
            fill_value=fill_value,
        )

    def rmod(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "rmod", rlocal_op("rmod"), axis=axis, fill_value=fill_value
        )

    def rpow(
        self, other: "DataFrame", axis: AxisType = 1, fill_value=None
    ) -> "DataFrame":
        return self._dispatch_binary_op(
            other, "rpow", rlocal_op("rpow"), axis=axis, fill_value=fill_value
        )
