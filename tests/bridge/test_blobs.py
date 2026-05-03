import os

import numpy as np
import pytest

from .arrays import BridgeError, np_to_npy_bytes, npy_bytes_to_np


# ---- tier 1: pure transport (no NumDot involvement) ----

def test_echo_blob_empty(bridge):
	header, blobs = bridge.call("echo_blob", blobs=[b""])
	assert header["ok"] is True
	assert blobs == [b""]


def test_echo_blob_random_small(bridge):
	payload = os.urandom(1024)
	header, blobs = bridge.call("echo_blob", blobs=[payload])
	assert header["ok"] is True
	assert blobs == [payload]


def test_echo_blob_random_large(bridge):
	payload = os.urandom(1_048_576)  # 1 MiB
	header, blobs = bridge.call("echo_blob", blobs=[payload])
	assert header["ok"] is True
	assert blobs == [payload]


def test_echo_blob_multiple(bridge):
	payloads = [os.urandom(7), os.urandom(2048), os.urandom(1)]
	header, blobs = bridge.call("echo_blob", blobs=payloads)
	assert header["ok"] is True
	assert blobs == payloads


# ---- tier 2: NumDot serializer round-trip ----

ROUND_TRIP_DTYPES = [
	np.int8, np.int16, np.int32, np.int64,
	np.uint8, np.uint16, np.uint32, np.uint64,
	np.float32, np.float64,
	np.bool_,
]


@pytest.mark.parametrize("dtype", ROUND_TRIP_DTYPES, ids=lambda d: np.dtype(d).name)
def test_echo_array_dtypes(bridge, dtype):
	if dtype is np.bool_:
		arr = np.array([True, False, True, False, True], dtype=dtype)
	else:
		arr = np.arange(5, dtype=dtype)

	header, blobs = bridge.call("echo_array", blobs=[np_to_npy_bytes(arr)])
	assert header.get("ok") is True, header
	assert len(blobs) == 1
	out = npy_bytes_to_np(blobs[0])

	assert out.shape == arr.shape
	assert out.dtype == arr.dtype
	assert np.array_equal(out, arr)


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 4)], ids=lambda s: f"shape{s}")
def test_echo_array_shapes(bridge, shape):
	arr = np.arange(int(np.prod(shape)) if shape else 1, dtype=np.float64).reshape(shape)
	header, blobs = bridge.call("echo_array", blobs=[np_to_npy_bytes(arr)])
	assert header.get("ok") is True, header
	out = npy_bytes_to_np(blobs[0])
	assert out.shape == arr.shape
	assert out.dtype == arr.dtype
	assert np.array_equal(out, arr)


def test_echo_array_empty(bridge):
	arr = np.zeros((0,), dtype=np.float32)
	header, blobs = bridge.call("echo_array", blobs=[np_to_npy_bytes(arr)])
	assert header.get("ok") is True, header
	out = npy_bytes_to_np(blobs[0])
	assert out.shape == (0,)
	assert out.dtype == np.float32


# ---- tier 3: real op (add) ----

def test_add_int32(bridge):
	a = np.array([1, 2, 3, 4], dtype=np.int32)
	b = np.array([10, 20, 30, 40], dtype=np.int32)
	out = bridge.call_array_op("add", a, b)
	assert np.array_equal(out, a + b)


def test_add_float64(bridge):
	a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
	b = np.array([0.5, -1.5, 2.5, -3.5], dtype=np.float64)
	out = bridge.call_array_op("add", a, b)
	assert np.array_equal(out, a + b)


def test_add_broadcasting(bridge):
	a = np.arange(3, dtype=np.int32)
	b = np.arange(3, dtype=np.int32).reshape(3, 1)
	out = bridge.call_array_op("add", a, b)
	assert np.array_equal(out, a + b)


def test_add_shape_mismatch_does_not_kill_bridge(bridge):
	a = np.arange(4, dtype=np.int32)
	b = np.arange(5, dtype=np.int32)
	with pytest.raises(BridgeError):
		bridge.call_array_op("add", a, b)
	# Bridge should still be alive for follow-up calls.
	header, _ = bridge.call("ping")
	assert header == {"ok": True, "pong": True}
