"""NumDot exposed as an Array API namespace, via the test bridge.

Usage from pytest:
    NUMDOT_GODOT=/path/to/godot ARRAY_API_TESTS_MODULE=numdot_xp \
        pytest tests/array-api-tests/array_api_tests/

Importing this module spawns a long-lived Godot subprocess that hosts
NumDot. All array operations are marshalled through the bridge as .npy
blobs, so the carrier type for arrays in this namespace is just
``numpy.ndarray`` — the bridge round-trips arrays through numpy on the
Python side and through ``nd.load`` / ``nd.dumpb`` on the Godot side.
"""

from __future__ import annotations

import atexit
import math
import os
import pathlib
import tempfile
from typing import Optional

import numpy as np

from bridge.arrays import BridgeError
from bridge.client import BridgeClient


__array_api_version__ = "2023.12"

# Carrier type. Subclass of np.ndarray so isinstance checks pass and the bridge
# encoder (bridge/client.py: isinstance arg, np.ndarray) accepts it transparently.
# We override the operator dunders below to dispatch back through nd, so that
# `x + y` actually exercises NumDot instead of silently going through numpy.
class ndarray(np.ndarray):
	__hash__ = None  # arrays are unhashable, matching np.ndarray

# Dtype objects. The Array API spec requires module-level attributes for each
# supported dtype. We expose numpy dtype scalar types directly — they compare
# equal to the dtypes attached to results coming back from the bridge.
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64
float32 = np.float32
float64 = np.float64
bool = np.bool_  # noqa: A001 — required spec name shadows the builtin
complex64 = np.complex64
complex128 = np.complex128

# Constants required by the spec.
e = math.e
pi = math.pi
inf = math.inf
nan = math.nan
newaxis = None

# Spec-required dtype introspection. The values returned by these functions
# are properties of the dtype itself (epsilon, smallest_normal, etc.) and are
# identical to numpy's, so we delegate.
finfo = np.finfo
iinfo = np.iinfo

# Spec-required namespace info. Returns capabilities/default_dtypes/devices/
# dtypes — all properties of the dtype space itself, which we share with
# numpy. Delegate. (Without this, array-api-tests's test_inspection_functions
# can't even collect, and Hypothesis warns "Could not determine whether
# module numdot_xp is an Array API library".)
__array_namespace_info__ = np.__array_namespace_info__


# --- bridge bootstrapping ----------------------------------------------------

_client: Optional[BridgeClient] = None
_log_path: Optional[pathlib.Path] = None


def _ensure_client() -> BridgeClient:
	global _client, _log_path
	if _client is not None and _client.is_alive():
		return _client
	if _client is not None:
		# Dead from a prior call — respawn before continuing.
		_client.respawn()
		return _client

	godot = os.environ.get("NUMDOT_GODOT")
	if not godot:
		raise RuntimeError(
			"NUMDOT_GODOT environment variable must point at the Godot binary "
			"used to run the bridge."
		)
	_log_path = pathlib.Path(tempfile.mkstemp(prefix="numdot_xp_", suffix=".log")[1])
	_client = BridgeClient(
		godot_binary=pathlib.Path(godot),
		log_path=_log_path,
		accept_timeout=10.0,
		call_timeout=30.0,
	)
	_client.__enter__()
	atexit.register(_atexit_cleanup)
	return _client


def _atexit_cleanup() -> None:
	global _client
	if _client is not None:
		try:
			_client.__exit__(None, None, None)
		except Exception:
			pass
		_client = None


def _call(func: str, *args):
	"""Adapter-side wrapper around BridgeClient.call_nd with respawn-on-death.

	If the bridge died on the previous call (Godot crashed), respawn before
	this call. We don't auto-retry the *current* call on failure — that would
	mask flaky failures from the test suite.

	Results are viewed as our ``ndarray`` subclass so subsequent operators on
	them dispatch through nd, not raw numpy.
	"""
	client = _ensure_client()
	return client.call_nd(func, *args).view(ndarray)


# --- operator dispatch -------------------------------------------------------
# Map each Python dunder to the nd function name it should call. Source of truth
# for op→func mapping is array_api_tests.dtype_helpers.op_to_func; we mirror it
# here to avoid importing the test suite into the adapter.

_BINARY_DUNDERS = {
	"__add__":      "add",
	"__sub__":      "subtract",
	"__mul__":      "multiply",
	"__truediv__":  "divide",
	"__floordiv__": "floor_divide",   # nd lacks this; tests will surface as failures.
	"__mod__":      "remainder",
	"__pow__":      "pow",
	"__matmul__":   "matmul",
	"__and__":      "bitwise_and",
	"__or__":       "bitwise_or",
	"__xor__":      "bitwise_xor",
	"__lshift__":   "bitwise_left_shift",
	"__rshift__":   "bitwise_right_shift",
	"__lt__":       "less",
	"__le__":       "less_equal",
	"__gt__":       "greater",
	"__ge__":       "greater_equal",
	"__eq__":       "equal",
	"__ne__":       "not_equal",
}

_UNARY_DUNDERS = {
	"__neg__":    "negative",
	"__pos__":    "positive",
	"__invert__": "bitwise_not",
	"__abs__":    "abs",
}

# In-place: just delegate to the non-in-place operation. The test harness only
# checks the post-`exec` value of the LHS (test_operators_and_elementwise_functions.py
# uses `xp.asarray(l, copy=True)` for `x1`, then `exec("x1 += x2")`), and Python
# rebinds the name to whatever __iadd__ returns — so true in-place semantics
# aren't required for correctness here.
_INPLACE_DUNDERS = {
	"__iadd__":      "add",
	"__isub__":      "subtract",
	"__imul__":      "multiply",
	"__itruediv__":  "divide",
	"__ifloordiv__": "floor_divide",
	"__imod__":      "remainder",
	"__ipow__":      "pow",
	"__imatmul__":   "matmul",
	"__iand__":      "bitwise_and",
	"__ior__":       "bitwise_or",
	"__ixor__":      "bitwise_xor",
	"__ilshift__":   "bitwise_left_shift",
	"__irshift__":   "bitwise_right_shift",
}


def _make_binary(nd_name):
	def dunder(self, other):
		return _call(nd_name, self, other)
	dunder.__name__ = nd_name
	return dunder


def _make_unary(nd_name):
	def dunder(self):
		return _call(nd_name, self)
	dunder.__name__ = nd_name
	return dunder


for _dunder_name, _nd_name in {**_BINARY_DUNDERS, **_INPLACE_DUNDERS}.items():
	setattr(ndarray, _dunder_name, _make_binary(_nd_name))
for _dunder_name, _nd_name in _UNARY_DUNDERS.items():
	setattr(ndarray, _dunder_name, _make_unary(_nd_name))
del _dunder_name, _nd_name


# Re-export the function shims at module level.
from ._funcs import *  # noqa: E402, F401, F403
