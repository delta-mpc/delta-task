import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar


def local_op(name: str):
    def _op(left, right, **kwargs):
        return getattr(left, name)(right, **kwargs)

    return _op


def rlocal_op(name: str):
    def _op(left, right, **kwargs):
        return getattr(right, name)(left, **kwargs)

    return _op


T = TypeVar("T")

class OpMixin(ABC, Generic[T]):
    @abstractmethod
    def _arith_method(self, other, name: str, op: Callable[[Any, Any], Any]) -> T:
        ...

    def __add__(self, other) -> T:
        return self._arith_method(other, "__add__", operator.add)

    def __radd__(self, other) -> T:
        def radd(left, right):
            return right + left

        return self._arith_method(other, "__radd__", radd)

    def __sub__(self, other) -> T:
        return self._arith_method(other, "__sub__", operator.sub)

    def __rsub__(self, other) -> T:
        def rsub(left, right):
            return right - left

        return self._arith_method(other, "__rsub__", rsub)

    def __mul__(self, other) -> T:
        return self._arith_method(other, "__mul__", operator.mul)

    def __rmul__(self, other) -> T:
        def rmul(left, right):
            return right * left

        return self._arith_method(other, "__rmul__", rmul)

    def __truediv__(self, other) -> T:
        return self._arith_method(other, "__truediv__", operator.truediv)

    def __rtruediv__(self, other) -> T:
        def rtruediv(left, right):
            return right / left

        return self._arith_method(other, "__rtruediv__", rtruediv)

    def __floordiv__(self, other) -> T:
        return self._arith_method(other, "__floordiv__", operator.floordiv)

    def __rfloordiv__(self, other) -> T:
        def rfloordiv(left, right):
            return right // left

        return self._arith_method(other, "__rfloordiv__", rfloordiv)

    def __mod__(self, other) -> T:
        return self._arith_method(other, "__mod__", operator.mod)

    def __rmod__(self, other) -> T:
        def rmod(left, right):
            if isinstance(right, str):
                typ = type(left).__name__
                raise TypeError(f"{typ} cannot perform the operation mod")

            return right % left

        return self._arith_method(other, "__rmod__", rmod)

    def __divmod__(self, other) -> T:
        return self._arith_method(other, "divmod", divmod)

    def __rdivmod__(self, other) -> T:
        def rdivmod(left, right):
            return divmod(right, left)

        return self._arith_method(other, "__rdivmod__", rdivmod)

    def __pow__(self, other) -> T:
        return self._arith_method(other, "__pow__", operator.pow)

    def __rpow__(self, other) -> T:
        def rpow(left, right):
            return right**left

        return self._arith_method(other, "__rpow__", rpow)
