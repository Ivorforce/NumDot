import io

import numpy as np


NUMPY_TO_ND_NAME: dict[np.dtype, str] = {
	np.dtype(np.int8): "Int8",
	np.dtype(np.int16): "Int16",
	np.dtype(np.int32): "Int32",
	np.dtype(np.int64): "Int64",
	np.dtype(np.uint8): "UInt8",
	np.dtype(np.uint16): "UInt16",
	np.dtype(np.uint32): "UInt32",
	np.dtype(np.uint64): "UInt64",
	np.dtype(np.float32): "Float32",
	np.dtype(np.float64): "Float64",
	np.dtype(np.bool_): "Bool",
	np.dtype(np.complex64): "Complex64",
	np.dtype(np.complex128): "Complex128",
}


class NdDtype:
	"""Sentinel for explicitly tagging a NumDot dtype name on the wire.

	Use when you want to pass a dtype that doesn't correspond to a numpy
	dtype object (rare). Numpy dtype objects and scalar types are
	auto-converted by the encoder, so most callers don't need this.
	"""

	def __init__(self, name: str):
		if name not in NUMPY_TO_ND_NAME.values():
			raise ValueError(f"unknown NumDot dtype name: {name}")
		self.name = name


class BridgeError(Exception):
	def __init__(self, header: dict):
		self.header = header
		super().__init__(f"bridge op failed: {header}")


def np_to_npy_bytes(arr: np.ndarray) -> bytes:
	buf = io.BytesIO()
	np.save(buf, arr, allow_pickle=False)
	return buf.getvalue()


def npy_bytes_to_np(blob: bytes) -> np.ndarray:
	return np.load(io.BytesIO(blob), allow_pickle=False)
