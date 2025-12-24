"""
# TODO Cythonize things. Improve performance.
# TODO Add tests
"""

import numpy as np
from typing import Union


def _is_number(e):
    try:
        f = float(e)
        return True
    except ValueError: # for malformed strings
        return False
    except TypeError: # for other types (Variable, Expr)
        return False


def _matrixexpr_richcmp(self, other, op):
    def _richcmp(self, other, op):
        if op == 1: # <=
            return self.__le__(other)
        elif op == 5: # >=
            return self.__ge__(other)
        elif op == 2: # ==
            return self.__eq__(other)
        else:
            raise NotImplementedError("Can only support constraints with '<=', '>=', or '=='.")

    if _is_number(other) or isinstance(other, Expr):
        res = np.empty(self.shape, dtype=object)
        res.flat = [_richcmp(i, other, op) for i in self.flat]

    elif isinstance(other, np.ndarray):
        out = np.broadcast(self, other)
        res = np.empty(out.shape, dtype=object)
        res.flat = [_richcmp(i, j, op) for i, j in out]

    else:
        raise TypeError(f"Unsupported type {type(other)}")

    return res.view(MatrixExprCons)


cdef class MatrixBase(np.ndarray):

    def __array_wrap__(self, array, context=None, return_scalar=False):
        res = super().__array_wrap__(array, context, return_scalar)
        if return_scalar and isinstance(res, np.ndarray) and res.ndim == 0:
            return res.item()
        elif isinstance(res, np.ndarray):
            return res.view(MatrixExpr)
        return res

    def sum(self, **kwargs):
        """
        Based on `numpy.ndarray.sum`, but returns a scalar if `axis=None`.
        This is useful for matrix expressions to compare with a matrix or a scalar.
        """

        if kwargs.get("axis") is None:
            # Speed up `.sum()` #1070
            return quicksum(self.flat)
        return super().sum(**kwargs).view(MatrixExpr)

    def __le__(self, other: Union[float, int, "Expr", np.ndarray, "MatrixExpr"]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 1)

    def __ge__(self, other: Union[float, int, "Expr", np.ndarray, "MatrixExpr"]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 5)

    def __eq__(self, other: Union[float, int, "Expr", np.ndarray, "MatrixExpr"]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 2)

    def _evaluate(self, SCIP* scip, SCIP_SOL* sol):
        res = np.zeros(self.shape, dtype=np.float64)
        res.flat = [i._evaluate(scip, sol) for i in self.flat]
        return res


class MatrixExpr(MatrixBase):
    ...


class MatrixExprCons(np.ndarray):

    def __le__(self, other: Union[float, int, np.ndarray]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 1)

    def __ge__(self, other: Union[float, int, np.ndarray]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 5)

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare MatrixExprCons with '=='.")
