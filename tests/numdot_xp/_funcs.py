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

from . import _call, ndarray


__all__ = [
	# creation
	"asarray", "zeros", "ones", "full", "arange", "empty", "eye", "linspace",
	"full_like", "ones_like", "zeros_like", "empty_like", "meshgrid",
	# elementwise (binary)
	"add", "subtract", "multiply", "divide", "floor_divide", "pow", "remainder",
	"equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
	"logical_and", "logical_or", "logical_xor",
	"bitwise_and", "bitwise_or", "bitwise_xor",
	"bitwise_left_shift", "bitwise_right_shift",
	"maximum", "minimum", "atan2", "hypot", "copysign", "logaddexp",
	# elementwise (unary)
	"negative", "positive", "abs", "sqrt", "square", "exp", "expm1", "log",
	"log2", "log10", "log1p",
	"sin", "cos", "tan", "asin", "acos", "atan",
	"sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
	"floor", "ceil", "round", "trunc", "sign", "signbit",
	"logical_not", "bitwise_invert",
	"isnan", "isfinite", "isinf",
	"conj", "real", "imag",
	"clip",
	# reductions
	"all", "any", "sum", "prod", "max", "min", "mean", "std", "var",
	"cumulative_sum", "cumulative_prod",
	# manipulation
	"reshape", "broadcast_to", "broadcast_arrays",
	"concat", "stack", "unstack", "flip", "moveaxis", "permute_dims",
	"squeeze", "tile", "expand_dims",
	# utility
	"diff",
	# selection
	"where",
	# dtype
	"astype", "can_cast", "result_type", "isdtype",
]


# ---- creation ---------------------------------------------------------------

def asarray(obj, /, *, dtype=None, device=None, copy=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	import numpy as np
	arr = np.asarray(obj, dtype=dtype)
	# NumDot exposes 'array' (not 'asarray'). We always round-trip through the
	# bridge so the returned object behaves identically to results from other ops.
	return _call("array", arr) if dtype is None else _call("array", arr, dtype)


def zeros(shape, *, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if dtype is None:
		dtype = np.float64  # spec default
	return _call("zeros", _shape(shape), dtype)


def ones(shape, *, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if dtype is None:
		dtype = np.float64
	return _call("ones", _shape(shape), dtype)


def full(shape, fill_value, *, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if dtype is None:
		# Spec: dtype inferred from fill_value — bool/int/float/complex map
		# to the corresponding default kind.
		if isinstance(fill_value, bool):
			dtype = np.bool_
		elif isinstance(fill_value, int):
			dtype = np.int64
		elif isinstance(fill_value, complex):
			dtype = np.complex128
		else:
			dtype = np.float64
	return _call("full", _shape(shape), fill_value, dtype)


def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
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
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if dtype is None:
		dtype = np.float64
	return _call("empty", _shape(shape), dtype)


def eye(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if n_cols is None:
		n_cols = n_rows
	if dtype is None:
		dtype = np.float64
	# NumDot signature: nd.eye(shape, k, dtype) — shape is a 2-element list,
	# not separate n_rows/n_cols positionals.
	return _call("eye", [n_rows, n_cols], k, dtype)


def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if dtype is None:
		dtype = np.float64
	# NumDot signature: nd.linspace(start, stop, num, endpoint, dtype) —
	# endpoint comes BEFORE dtype.
	return _call("linspace", start, stop, num, endpoint, dtype)


def full_like(x, /, fill_value, *, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if dtype is None:
		return _call("full_like", x, fill_value)
	return _call("full_like", x, fill_value, dtype)


def ones_like(x, /, *, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if dtype is None:
		return _call("ones_like", x)
	return _call("ones_like", x, dtype)


def zeros_like(x, /, *, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if dtype is None:
		return _call("zeros_like", x)
	return _call("zeros_like", x, dtype)


def empty_like(x, /, *, dtype=None, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
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
def floor_divide(x1, x2, /):     return _call("floor_divide", x1, x2)
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
def atan2(x1, x2, /):            return _call("atan2", x1, x2)
def hypot(x1, x2, /):            return _call("hypot", x1, x2)
def copysign(x1, x2, /):         return _call("copysign", x1, x2)
def logaddexp(x1, x2, /):        return _call("logaddexp", x1, x2)


# ---- elementwise unary ------------------------------------------------------

def negative(x, /):       return _call("negative", x)
def positive(x, /):       return _call("positive", x)
def abs(x, /):            return _call("abs", x)  # noqa: A001
def sqrt(x, /):           return _call("sqrt", x)
def square(x, /):         return _call("square", x)
def exp(x, /):            return _call("exp", x)
def expm1(x, /):          return _call("expm1", x)
def log(x, /):            return _call("log", x)
def log2(x, /):           return _call("log2", x)
def log10(x, /):          return _call("log10", x)
def log1p(x, /):          return _call("log1p", x)
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
def round(x, /):          return _call("round", x)         # noqa: A001
def trunc(x, /):          return _call("trunc", x)
def sign(x, /):           return _call("sign", x)
def signbit(x, /):        return _call("signbit", x)
def logical_not(x, /):    return _call("logical_not", x)
def bitwise_invert(x, /): return _call("bitwise_not", x)  # spec name -> nd name
def isnan(x, /):          return _call("is_nan", x)       # spec name -> nd name
def isfinite(x, /):       return _call("is_finite", x)
def isinf(x, /):          return _call("is_inf", x)
def conj(x, /):           return _call("conjugate", x)    # spec name -> nd name
def real(x, /):           return _call("real", x)
def imag(x, /):           return _call("imag", x)


def clip(x, /, min=None, max=None):  # noqa: A001 — spec name shadows builtin
	# nd.clip takes (a, min, max) positionally; min/max may be scalars or arrays.
	return _call("clip", x, min, max)


# ---- reductions -------------------------------------------------------------
# All NumDot reductions share one signature: nd.<r>(a, axes=null), where axes
# is int | list[int] | null. keepdims and dtype are not native — emulated here.

def _normalize_axes(axis, ndim):
	"""Array API axis kwarg → list of normalized non-negative axes (or None for full)."""
	if axis is None:
		return None
	if isinstance(axis, int):
		axes = (axis,)
	else:
		axes = tuple(axis)
	return [a % ndim for a in axes]


def _keepdims_shape(in_shape, norm_axes):
	"""Reduced-with-keepdims shape: in_shape with each reduced axis → 1."""
	if norm_axes is None:
		return [1] * len(in_shape)
	out = list(in_shape)
	for a in norm_axes:
		out[a] = 1
	return out


def _reduced_size(in_shape, axis):
	"""Number of elements collapsed by the reduction (for std/var correction)."""
	norm = _normalize_axes(axis, len(in_shape))
	if norm is None:
		n = 1
		for s in in_shape:
			n *= s
		return n
	n = 1
	for a in norm:
		n *= in_shape[a]
	return n


def _reduce(nd_func, x, axis, keepdims):
	norm = _normalize_axes(axis, x.ndim)
	if norm is None:
		out = _call(nd_func, x)
	elif len(norm) == 1:
		out = _call(nd_func, x, norm[0])
	else:
		out = _call(nd_func, x, norm)
	if keepdims:
		out = _call("reshape", out, _keepdims_shape(x.shape, norm))
	return out


def all(x, /, *, axis=None, keepdims=False):  # noqa: A001
	return _reduce("all", x, axis, keepdims)


def any(x, /, *, axis=None, keepdims=False):
	return _reduce("any", x, axis, keepdims)


def max(x, /, *, axis=None, keepdims=False):  # noqa: A001
	return _reduce("max", x, axis, keepdims)


def min(x, /, *, axis=None, keepdims=False):  # noqa: A001
	return _reduce("min", x, axis, keepdims)


def mean(x, /, *, axis=None, keepdims=False):
	return _reduce("mean", x, axis, keepdims)


def sum(x, /, *, axis=None, dtype=None, keepdims=False):  # noqa: A001
	out = _reduce("sum", x, axis, keepdims)
	# nd.sum's accumulator dtype follows numpy's "promote small ints to int64"
	# rule, so an explicit narrower dtype only takes effect if we cast the
	# output. Doing it after the reduction also avoids `nd.array(x, dtype)`
	# truncating in-range values (e.g. uint8 sum into int8) before summing.
	if dtype is not None:
		out = _call("array", out, dtype)
	return out


def prod(x, /, *, axis=None, dtype=None, keepdims=False):
	out = _reduce("prod", x, axis, keepdims)
	if dtype is not None:
		out = _call("array", out, dtype)
	return out


def _cumulative(nd_func, x, axis, dtype, include_initial, identity):
	if axis is None:
		out = _call(nd_func, x)
		axis_norm = 0
	else:
		axis_norm = axis if axis >= 0 else axis + x.ndim
		out = _call(nd_func, x, axis)
	# Post-cast (mirrors sum/prod): nd's accumulator dtype already promotes
	# narrow ints to int64; casting before would re-introduce overflow.
	if dtype is not None:
		out = _call("array", out, dtype)
	if include_initial:
		prefix_shape = list(out.shape)
		prefix_shape[axis_norm] = 1
		prefix = np.full(prefix_shape, identity, dtype=out.dtype)
		out = _call("concatenate", [prefix, out], axis_norm)
	return out


def cumulative_sum(x, /, *, axis=None, dtype=None, include_initial=False):
	return _cumulative("cumsum", x, axis, dtype, include_initial, identity=0)


def cumulative_prod(x, /, *, axis=None, dtype=None, include_initial=False):
	return _cumulative("cumprod", x, axis, dtype, include_initial, identity=1)


def diff(x, /, *, axis=-1, n=1, prepend=None, append=None):
	if prepend is not None or append is not None:
		parts = []
		if prepend is not None:
			parts.append(prepend)
		parts.append(x)
		if append is not None:
			parts.append(append)
		x = _call("concatenate", parts, axis)
	return _call("diff", x, n, axis)


def var(x, /, *, axis=None, correction=0.0, keepdims=False):
	out = _reduce("var", x, axis, keepdims)
	if correction != 0.0:
		n = _reduced_size(x.shape, axis)
		if n - correction > 0:
			# Cast the factor to x's dtype so multiply doesn't promote float32 → float64.
			factor = np.asarray(n / (n - correction), dtype=x.dtype)
			out = _call("multiply", out, factor)
	return out


def std(x, /, *, axis=None, correction=0.0, keepdims=False):
	if correction == 0.0:
		return _reduce("std", x, axis, keepdims)
	return _call("sqrt", var(x, axis=axis, correction=correction, keepdims=keepdims))


# ---- manipulation -----------------------------------------------------------

def reshape(x, /, shape, *, copy=None):
	if copy is False:
		raise NotImplementedError("reshape with copy=False not yet wired")
	return _call("reshape", x, _shape(shape))


def broadcast_to(x, /, shape):
	return _call("broadcast_to", x, _shape(shape))


def broadcast_arrays(*arrays):
	# Array API requires a list of broadcast views. nd has broadcast_to but
	# no broadcast_shapes; use numpy to derive the common shape, then route
	# each input through nd.broadcast_to.
	target = np.broadcast_shapes(*[a.shape for a in arrays])
	return [_call("broadcast_to", a, list(target)) for a in arrays]


def concat(arrays, /, *, axis=0):
	# nd.concatenate handles axis=None natively (flattens each input first).
	return _call("concatenate", list(arrays), axis)


def stack(arrays, /, *, axis=0):
	# nd.stack is moveaxis on a single array, not numpy stack — compose
	# Array-API stack semantics from reshape (insert a length-1 axis at the
	# requested position) + concatenate along that axis.
	arrays = [np.asarray(a) for a in arrays]
	if not arrays:
		raise ValueError("stack: arrays must be non-empty")
	out_ndim = arrays[0].ndim + 1
	norm_axis = axis if axis >= 0 else axis + out_ndim
	new_shape = list(arrays[0].shape)
	new_shape.insert(norm_axis, 1)
	expanded = [_call("reshape", a, new_shape) for a in arrays]
	return _call("concatenate", expanded, norm_axis)


def unstack(x, /, *, axis=0):
	# Array API requires a tuple. nd.unstack would need bridge support for
	# returning a list of arrays (only one blob comes back today), so we do
	# the split client-side via numpy.
	return tuple(arr.view(ndarray) for arr in np.moveaxis(np.asarray(x), axis, 0))


def meshgrid(*arrays, indexing="xy"):
	# nd.meshgrid exists on the C++ side, but the bridge can't return a list of
	# arrays in one call (same limitation as unstack/split). Build each output
	# independently: reshape to broadcast-shape, then nd.broadcast_to.
	if not arrays:
		return []
	n = len(arrays)
	if builtins.any(a.ndim != 1 for a in arrays):
		raise ValueError("meshgrid: each input must be 1-D")
	xy = (indexing == "xy") and n >= 2
	def axis_of(i):
		if xy and i == 0: return 1
		if xy and i == 1: return 0
		return i
	out_shape = [None] * n
	for i, a in enumerate(arrays):
		out_shape[axis_of(i)] = a.shape[0]
	outs = []
	for i, a in enumerate(arrays):
		dims = [1] * n
		dims[axis_of(i)] = a.shape[0]
		reshaped = _call("reshape", a, list(dims))
		outs.append(_call("broadcast_to", reshaped, list(out_shape)))
	return outs


def flip(x, /, *, axis=None):
	# nd.flip is single-axis; iterate for axis=None / tuples.
	if axis is None:
		axes = range(x.ndim)
	elif isinstance(axis, int):
		axes = (axis,)
	else:
		axes = tuple(axis)
	out = x
	for a in axes:
		out = _call("flip", out, a)
	return out


def moveaxis(x, source, destination, /):
	src = source if isinstance(source, int) else list(source)
	dst = destination if isinstance(destination, int) else list(destination)
	return _call("moveaxis", x, src, dst)


def permute_dims(x, /, axes):
	return _call("transpose", x, list(axes))


def expand_dims(x, /, axis=0):
	# Array-API spec: axis must lie in [-ndim-1, ndim], else IndexError.
	if axis < -x.ndim - 1 or axis > x.ndim:
		raise IndexError(f"expand_dims: axis {axis} out of range for ndim {x.ndim}")
	return _call("expand_dims", x, axis)


def squeeze(x, /, axis):
	# Pre-validate: spec mandates ValueError on non-1 axes, but nd reports
	# errors as null returns (→ BridgeError). Cheaper than re-typing the error.
	axes = (axis,) if isinstance(axis, int) else tuple(axis)
	for a in axes:
		if x.shape[a] != 1:
			raise ValueError(f"cannot squeeze axis {a} of shape {x.shape}")
	return _call("squeeze", x, axis if isinstance(axis, int) else list(axis))


def tile(x, repetitions, /):
	# nd.tile takes (v, reps, inner=False).
	return _call("tile", x, list(repetitions), False)


# ---- selection --------------------------------------------------------------

def where(condition, x1, x2, /):
	return _call("where", condition, x1, x2)


# ---- dtype --------------------------------------------------------------------
# Pure dtype/introspection — delegated to numpy. astype routes through nd so
# the actual conversion lives in the va layer.

def astype(x, dtype, /, *, copy=True, device=None):
	if device not in (None, "cpu"):
		raise ValueError(f"unsupported device: {device!r}")
	if not copy and x.dtype == dtype:
		return x
	return _call("array", x, dtype)


def can_cast(from_, to, /):
	return np.can_cast(from_, to)


def result_type(*arrays_and_dtypes):
	return np.result_type(*arrays_and_dtypes)


def isdtype(dtype, kind):
	return np.isdtype(dtype, kind)
