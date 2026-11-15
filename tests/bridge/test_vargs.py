"""Functions whose vfuncs take extra runtime args (vargs) beyond their arrays."""
import numpy as np


def test_is_close_atol(bridge):
	a = np.array([1.0, 1.0], dtype=np.float64)
	b = np.array([1.5, 3.0], dtype=np.float64)
	out = bridge.call_nd("is_close", a, b, 0.0, 1.0, False)
	assert np.array_equal(out, np.isclose(a, b, rtol=0.0, atol=1.0))


def test_is_close_rtol(bridge):
	a = np.array([100.0, 100.0], dtype=np.float64)
	b = np.array([101.0, 110.0], dtype=np.float64)
	out = bridge.call_nd("is_close", a, b, 0.05, 0.0, False)
	assert np.array_equal(out, np.isclose(a, b, rtol=0.05, atol=0.0))


def test_is_close_equal_nan(bridge):
	a = np.array([np.nan, 1.0], dtype=np.float64)
	b = np.array([np.nan, 1.0], dtype=np.float64)
	assert np.array_equal(bridge.call_nd("is_close", a, b, 0.0, 0.0, True), [True, True])
	assert np.array_equal(bridge.call_nd("is_close", a, b, 0.0, 0.0, False), [False, True])


def test_fft_axis(bridge):
	# xtensor-signal's FFT only supports power-of-2 lengths.
	x = np.arange(8, dtype=np.float64).reshape(2, 4)
	for axis in (0, 1, -1):
		out = bridge.call_nd("fft", x, axis)
		assert np.allclose(out, np.fft.fft(x, axis=axis))


def test_pad_constant(bridge):
	x = np.array([[1, 2], [3, 4]], dtype=np.int32)
	# Nested pad widths would cross the bridge as JSON floats; a flat
	# (before, after) pair is sent as ints and applies to every axis.
	out = bridge.call_nd("pad", x, [1, 2], 0, 7)
	assert np.array_equal(out, np.pad(x, (1, 2), constant_values=7))
