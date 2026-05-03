import os
import pathlib
import socket
import subprocess
from typing import Optional, Sequence

import numpy as np

from .arrays import (
	NUMPY_TO_ND_NAME,
	BridgeError,
	NdDtype,
	np_to_npy_bytes,
	npy_bytes_to_np,
)
from .protocol import encode_frame, read_frame


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEMO_PATH = REPO_ROOT / "demo"


def _encode_nd_args(args) -> tuple[list, list[bytes]]:
	specs: list = []
	blobs: list[bytes] = []
	for arg in args:
		if isinstance(arg, np.ndarray):
			specs.append({"$blob": len(blobs)})
			blobs.append(np_to_npy_bytes(arg))
		elif isinstance(arg, NdDtype):
			specs.append({"$dtype": arg.name})
		elif isinstance(arg, np.dtype):
			specs.append({"$dtype": NUMPY_TO_ND_NAME[arg]})
		elif isinstance(arg, type) and issubclass(arg, np.generic):
			specs.append({"$dtype": NUMPY_TO_ND_NAME[np.dtype(arg)]})
		elif arg is None:
			specs.append({"$null": True})
		elif isinstance(arg, bool):
			# Order matters: bool is an int subclass, must be checked first.
			specs.append({"$value": arg})
		elif isinstance(arg, int):
			# JSON parsing in GDScript turns every number into a float64,
			# losing precision for ints with |value| > 2**53. Send as a
			# string; GDScript's int() coerces strings transparently.
			specs.append({"$int": str(arg)})
		elif isinstance(arg, (list, tuple)) and arg and all(
			isinstance(x, int) and not isinstance(x, bool) for x in arg
		):
			specs.append({"$ints": [str(x) for x in arg]})
		else:
			specs.append({"$value": arg})
	return specs, blobs


class BridgeClient:
	def __init__(
		self,
		godot_binary: pathlib.Path,
		log_path: pathlib.Path,
		accept_timeout: float = 5.0,
		call_timeout: float = 5.0,
	):
		self.godot_binary = pathlib.Path(godot_binary)
		self.log_path = pathlib.Path(log_path)
		self.accept_timeout = accept_timeout
		self.call_timeout = call_timeout

		self._listener: Optional[socket.socket] = None
		self._sock: Optional[socket.socket] = None
		self._process: Optional[subprocess.Popen] = None
		self._log_file = None
		self._dead = False

	def __enter__(self) -> "BridgeClient":
		self._spawn()
		return self

	def is_alive(self) -> bool:
		return (
			not self._dead
			and self._sock is not None
			and self._process is not None
			and self._process.poll() is None
		)

	def respawn(self) -> None:
		"""Tear down a dead bridge and spawn a fresh one. Used by long-lived
		callers (the array-api adapter) to recover after Godot crashes."""
		self._teardown()
		self._spawn()

	def _spawn(self) -> None:
		self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._listener.bind(("127.0.0.1", 0))
		self._listener.listen(1)
		self._listener.settimeout(self.accept_timeout)
		port = self._listener.getsockname()[1]

		self._log_file = open(self.log_path, "ab")
		env = {**os.environ, "NUMDOT_BRIDGE_PORT": str(port)}
		self._process = subprocess.Popen(
			[
				str(self.godot_binary),
				"--path", str(DEMO_PATH),
				"--headless",
				"res://tests/bridge.tscn",
			],
			env=env,
			stdout=self._log_file,
			stderr=self._log_file,
		)

		try:
			self._sock, _ = self._listener.accept()
		except socket.timeout:
			self._kill_process()
			raise TimeoutError(
				f"Godot did not connect within {self.accept_timeout}s. "
				f"See log: {self.log_path}"
			)
		finally:
			self._listener.close()
			self._listener = None

		self._sock.settimeout(self.call_timeout)
		self._dead = False

	def call(self, op: str, blobs: Sequence[bytes] = (), **kwargs) -> tuple[dict, list[bytes]]:
		if self._sock is None or self._dead:
			raise RuntimeError("BridgeClient is not active")
		header = {"op": op, **kwargs}
		try:
			self._sock.sendall(encode_frame(header, blobs))
			return read_frame(self._sock)
		except (ConnectionError, OSError, socket.timeout):
			self._dead = True
			raise

	def call_nd(self, func: str, *args) -> np.ndarray:
		arg_specs, blobs = _encode_nd_args(args)
		header, out_blobs = self.call("nd_call", blobs=blobs, func=func, args=arg_specs)
		if not header.get("ok"):
			raise BridgeError(header)
		if len(out_blobs) != 1:
			raise RuntimeError(f"expected 1 output blob, got {len(out_blobs)}")
		return npy_bytes_to_np(out_blobs[0])

	def call_array_op(self, func: str, *arrays: np.ndarray) -> np.ndarray:
		# Backwards-compat shim: previously a hand-written op on the bridge,
		# now routed through nd_call.
		return self.call_nd(func, *arrays)

	def __exit__(self, exc_type, exc, tb) -> None:
		self._teardown(graceful=exc_type is None)

	def _teardown(self, graceful: bool = True) -> None:
		try:
			if graceful and self._sock is not None and not self._dead:
				try:
					self.call("shutdown")
				except (OSError, ConnectionError, RuntimeError):
					pass
		finally:
			if self._sock is not None:
				try:
					self._sock.close()
				except OSError:
					pass
				self._sock = None

			if self._process is not None:
				try:
					self._process.wait(timeout=2.0)
				except subprocess.TimeoutExpired:
					self._kill_process()
				self._process = None

			if self._log_file is not None:
				self._log_file.close()
				self._log_file = None

			self._dead = True

	def _kill_process(self) -> None:
		if self._process is None:
			return
		self._process.terminate()
		try:
			self._process.wait(timeout=1.0)
		except subprocess.TimeoutExpired:
			self._process.kill()
			self._process.wait()
