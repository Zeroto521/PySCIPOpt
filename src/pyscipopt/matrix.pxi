"""
# TODO Cythonize things. Improve performance.
# TODO Add tests
"""

from numbers import Number
from typing import Union

import numpy as np

from cpython.object cimport Py_EQ, Py_LE, Py_GE


cdef class MatrixExprCons(np.ndarray):

    @staticmethod
    cdef MatrixExprCons _cmp(self, other, op):
        if isinstance(other, (Number, Expr)):
            res = np.empty(self.shape, dtype=object)
            res.flat[:] = [op(i, other) for i in self.flat]
        elif isinstance(other, np.ndarray):
            out = np.broadcast(self, other)
            res = np.empty(out.shape, dtype=object)
            res.flat[:] = [op(i, j) for i, j in out]
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return res.view(MatrixExprCons)

    cdef MatrixExprCons __richcmp__(self, float other, int op):
        if op == Py_LE:
            return MatrixExprCons._cmp(self, other, lambda x, y: x <= y)
        elif op == Py_GE:
            return MatrixExprCons._cmp(self, other, lambda x, y: x >= y)
        elif op == Py_EQ:
            raise NotImplementedError("Cannot compare MatrixExprCons with '=='.")
        raise NotImplementedError("Can only compare MatrixExprCons with '<=' or '>='.")


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

    cdef MatrixExprCons __richcmp__(self, other: Union[Number, Expr, np.ndarray, MatrixExpr], int op):
        if op == Py_LE:
            return MatrixExprCons._cmp(self, other, lambda x, y: x <= y)
        elif op == Py_GE:
            return MatrixExprCons._cmp(self, other, lambda x, y: x >= y)
        elif op == Py_EQ:
            return MatrixExprCons._cmp(self, other, lambda x, y: x >= y)
        raise NotImplementedError("Can only compare MatrixExprCons with '<=', '>=' or '=='.")

    def _evaluate(self, SCIP* scip, SCIP_SOL* sol):
        res = np.zeros(self.shape, dtype=np.float64)
        res.flat = [i._evaluate(scip, sol) for i in self.flat]
        return res


class MatrixExpr(MatrixBase):
    ...
