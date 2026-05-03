import io

import numpy as np


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
