"""Array API function shims that translate to NumDot calls.

Each function does the minimum signature translation, then delegates to
the bridge. Where NumDot's positional-arg convention diverges from the
Array API spec (which is mostly keyword-arg-heavy), the shim does the
rearranging; the bridge itself takes only positional args.

Functions are kept deliberately minimal at first — coverage will expand
by removing entries from xfails.txt and adding shims here as needed.
"""

from __future__ import annotations

import builtins  # the Array API shims below shadow several Python builtins
                 # (all, any, min, max, sum, pow, abs); use builtins.<name>
                 # inside this module if the real builtin is what you want.

import numpy as np

from . import _call


__all__ = [
	# creation
	"asarray", "zeros", "ones", "full", "arange", "empty", "eye", "linspace",
	"full_like", "ones_like", "zeros_like", "empty_like",
	# elementwise (binary)
	"add", "subtract", "multiply", "divide", "pow", "remainder",
	"equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
	"logical_and", "logical_or", "logical_xor",
	"bitwise_and", "bitwise_or", "bitwise_xor",
	"bitwise_left_shift", "bitwise_right_shift",
	"maximum", "minimum",
	# elementwise (unary)
	"negative", "positive", "abs", "sqrt", "square", "exp", "log",
	"sin", "cos", "tan", "asin", "acos", "atan",
	"sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
	"floor", "ceil", "sign",
	"logical_not", "bitwise_invert",
	"isnan", "isfinite", "isinf",
	# reductions
	"all", "any", "sum", "prod", "max", "min", "mean",
	# manipulation
	"reshape",
]


# ---- creation ---------------------------------------------------------------

def asarray(obj, /, *, dtype=None, device=None, copy=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	import numpy as np
	arr = np.asarray(obj, dtype=dtype)
	# NumDot exposes 'array' (not 'asarray'). We always round-trip through the
	# bridge so the returned object behaves identically to results from other ops.
	return _call("array", arr) if dtype is None else _call("array", arr, dtype)


def zeros(shape, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		dtype = np.float64  # spec default
	return _call("zeros", _shape(shape), dtype)


def ones(shape, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		dtype = np.float64
	return _call("ones", _shape(shape), dtype)


def full(shape, fill_value, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		dtype = np.float64
	return _call("full", _shape(shape), fill_value, dtype)


def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if stop is None:
		start, stop = 0, start
	if dtype is None:
		# Spec: dtype inferred from inputs — all-int → default int, else default float.
		if builtins.all(isinstance(v, int) and not isinstance(v, bool) for v in (start, stop, step)):
			dtype = np.int64
		else:
			dtype = np.float64
	return _call("arange", start, stop, step, dtype)


def empty(shape, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		dtype = np.float64
	return _call("empty", _shape(shape), dtype)


def eye(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if n_cols is None:
		n_cols = n_rows
	if dtype is None:
		dtype = np.float64
	return _call("eye", n_rows, n_cols, k, dtype)


def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		dtype = np.float64
	# NumDot's linspace signature is unverified — assume (start, stop, num, dtype, endpoint).
	# If wrong, this raises a clear BridgeError and we adjust.
	return _call("linspace", start, stop, num, dtype, endpoint)


def full_like(x, /, fill_value, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		return _call("full_like", x, fill_value)
	return _call("full_like", x, fill_value, dtype)


def ones_like(x, /, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		return _call("ones_like", x)
	return _call("ones_like", x, dtype)


def zeros_like(x, /, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		return _call("zeros_like", x)
	return _call("zeros_like", x, dtype)


def empty_like(x, /, *, dtype=None, device=None):
	if device is not None:
		raise ValueError("device kwarg is not supported")
	if dtype is None:
		return _call("empty_like", x)
	return _call("empty_like", x, dtype)


def _shape(shape):
	"""Normalise an Array API 'shape' arg to a list (single int → 1-d tuple)."""
	if isinstance(shape, int):
		return [shape]
	return list(shape)


# ---- elementwise binary -----------------------------------------------------

def add(x1, x2, /):              return _call("add", x1, x2)
def subtract(x1, x2, /):         return _call("subtract", x1, x2)
def multiply(x1, x2, /):         return _call("multiply", x1, x2)
def divide(x1, x2, /):           return _call("divide", x1, x2)
def pow(x1, x2, /):              return _call("pow", x1, x2)  # noqa: A001
def remainder(x1, x2, /):        return _call("remainder", x1, x2)
def equal(x1, x2, /):            return _call("equal", x1, x2)
def not_equal(x1, x2, /):        return _call("not_equal", x1, x2)
def less(x1, x2, /):             return _call("less", x1, x2)
def less_equal(x1, x2, /):       return _call("less_equal", x1, x2)
def greater(x1, x2, /):          return _call("greater", x1, x2)
def greater_equal(x1, x2, /):    return _call("greater_equal", x1, x2)
def logical_and(x1, x2, /):      return _call("logical_and", x1, x2)
def logical_or(x1, x2, /):       return _call("logical_or", x1, x2)
def logical_xor(x1, x2, /):      return _call("logical_xor", x1, x2)
def bitwise_and(x1, x2, /):      return _call("bitwise_and", x1, x2)
def bitwise_or(x1, x2, /):       return _call("bitwise_or", x1, x2)
def bitwise_xor(x1, x2, /):      return _call("bitwise_xor", x1, x2)
def bitwise_left_shift(x1, x2, /):  return _call("bitwise_left_shift", x1, x2)
def bitwise_right_shift(x1, x2, /): return _call("bitwise_right_shift", x1, x2)
def maximum(x1, x2, /):          return _call("maximum", x1, x2)
def minimum(x1, x2, /):          return _call("minimum", x1, x2)


# ---- elementwise unary ------------------------------------------------------

def negative(x, /):       return _call("negative", x)
def positive(x, /):       return _call("positive", x)
def abs(x, /):            return _call("abs", x)  # noqa: A001
def sqrt(x, /):           return _call("sqrt", x)
def square(x, /):         return _call("square", x)
def exp(x, /):            return _call("exp", x)
def log(x, /):            return _call("log", x)
def sin(x, /):            return _call("sin", x)
def cos(x, /):            return _call("cos", x)
def tan(x, /):            return _call("tan", x)
def asin(x, /):           return _call("asin", x)
def acos(x, /):           return _call("acos", x)
def atan(x, /):           return _call("atan", x)
def sinh(x, /):           return _call("sinh", x)
def cosh(x, /):           return _call("cosh", x)
def tanh(x, /):           return _call("tanh", x)
def asinh(x, /):          return _call("asinh", x)
def acosh(x, /):          return _call("acosh", x)
def atanh(x, /):          return _call("atanh", x)
def floor(x, /):          return _call("floor", x)
def ceil(x, /):           return _call("ceil", x)
def sign(x, /):           return _call("sign", x)
def logical_not(x, /):    return _call("logical_not", x)
def bitwise_invert(x, /): return _call("bitwise_not", x)  # spec name -> nd name
def isnan(x, /):          return _call("is_nan", x)       # spec name -> nd name
def isfinite(x, /):       return _call("is_finite", x)
def isinf(x, /):          return _call("is_inf", x)


# ---- reductions -------------------------------------------------------------
# Array API reductions take (x, *, axis=None, keepdims=False[, dtype=None]).
# NumDot's signatures may vary; for now we only handle the no-axis,
# default-keepdims path. axis/keepdims/dtype handling will be added once we
# know the matching NumDot positional-arg layout.

def all(x, /, *, axis=None, keepdims=False):  # noqa: A001
	if axis is not None or keepdims:
		raise NotImplementedError("axis/keepdims for all() not yet wired")
	return _call("all", x)


def any(x, /, *, axis=None, keepdims=False):
	if axis is not None or keepdims:
		raise NotImplementedError("axis/keepdims for any() not yet wired")
	return _call("any", x)


def sum(x, /, *, axis=None, dtype=None, keepdims=False):  # noqa: A001
	if axis is not None or keepdims or dtype is not None:
		raise NotImplementedError("axis/keepdims/dtype for sum() not yet wired")
	return _call("sum", x)


def prod(x, /, *, axis=None, dtype=None, keepdims=False):
	if axis is not None or keepdims or dtype is not None:
		raise NotImplementedError("axis/keepdims/dtype for prod() not yet wired")
	return _call("prod", x)


def max(x, /, *, axis=None, keepdims=False):  # noqa: A001
	if axis is not None or keepdims:
		raise NotImplementedError("axis/keepdims for max() not yet wired")
	return _call("max", x)


def min(x, /, *, axis=None, keepdims=False):  # noqa: A001
	if axis is not None or keepdims:
		raise NotImplementedError("axis/keepdims for min() not yet wired")
	return _call("min", x)


def mean(x, /, *, axis=None, keepdims=False):
	if axis is not None or keepdims:
		raise NotImplementedError("axis/keepdims for mean() not yet wired")
	return _call("mean", x)


# ---- manipulation -----------------------------------------------------------

def reshape(x, /, shape, *, copy=None):
	if copy is False:
		raise NotImplementedError("reshape with copy=False not yet wired")
	return _call("reshape", x, _shape(shape))
