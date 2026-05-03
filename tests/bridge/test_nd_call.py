import numpy as np
import pytest

from .arrays import BridgeError


# ---- functions with array args ----

def test_nd_call_add_int32(bridge):
	a = np.array([1, 2, 3, 4], dtype=np.int32)
	b = np.array([10, 20, 30, 40], dtype=np.int32)
	out = bridge.call_nd("add", a, b)
	assert out.dtype == np.int32
	assert np.array_equal(out, a + b)


def test_nd_call_add_float64(bridge):
	a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
	b = np.array([0.5, -1.5, 2.5, -3.5], dtype=np.float64)
	out = bridge.call_nd("add", a, b)
	assert out.dtype == np.float64
	assert np.array_equal(out, a + b)


def test_nd_call_sin(bridge):
	x = np.linspace(0.0, np.pi, 5, dtype=np.float64)
	out = bridge.call_nd("sin", x)
	assert out.dtype == np.float64
	assert np.allclose(out, np.sin(x))


# ---- shape + dtype creation functions ----

def test_nd_call_zeros(bridge):
	out = bridge.call_nd("zeros", [3, 4], np.float32)
	assert out.shape == (3, 4)
	assert out.dtype == np.float32
	assert np.all(out == 0)


def test_nd_call_ones_int64(bridge):
	out = bridge.call_nd("ones", [2, 3], np.int64)
	assert out.shape == (2, 3)
	assert out.dtype == np.int64
	assert np.all(out == 1)


# ---- mixed value args ----

def test_nd_call_arange(bridge):
	out = bridge.call_nd("arange", 0, 10, 1, np.int32)
	assert out.dtype == np.int32
	assert np.array_equal(out, np.arange(0, 10, 1, dtype=np.int32))


# ---- error paths (bridge stays alive) ----

def test_nd_call_unknown_function(bridge):
	with pytest.raises(BridgeError) as exc_info:
		bridge.call_nd("definitely_not_a_real_func", np.arange(3))
	assert exc_info.value.header["error"] == "op_failed"
	assert "unknown function" in exc_info.value.header["message"]
	# Bridge should still be alive.
	header, _ = bridge.call("ping")
	assert header == {"ok": True, "pong": True}


def test_nd_call_shape_mismatch(bridge):
	a = np.arange(3, dtype=np.int32)
	b = np.arange(4, dtype=np.int32)
	with pytest.raises(BridgeError):
		bridge.call_nd("add", a, b)
	# Bridge should still be alive.
	header, _ = bridge.call("ping")
	assert header == {"ok": True, "pong": True}
