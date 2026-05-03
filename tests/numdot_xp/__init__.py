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

# Carrier type. Spec consumers (e.g. hypothesis) check isinstance(x, ndarray);
# everything we hand back is a real numpy.ndarray produced by np.load on the
# bridge response, so this is honest.
ndarray = np.ndarray

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
	"""
	client = _ensure_client()
	return client.call_nd(func, *args)


# Re-export the function shims at module level.
from ._funcs import *  # noqa: E402, F401, F403
