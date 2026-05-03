import os
import pathlib
import socket
import subprocess
from typing import Optional

from .protocol import encode_frame, read_frame


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEMO_PATH = REPO_ROOT / "demo"


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

	def __enter__(self) -> "BridgeClient":
		self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._listener.bind(("127.0.0.1", 0))
		self._listener.listen(1)
		self._listener.settimeout(self.accept_timeout)
		port = self._listener.getsockname()[1]

		self._log_file = open(self.log_path, "wb")
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
		return self

	def call(self, op: str, **kwargs) -> dict:
		if self._sock is None:
			raise RuntimeError("BridgeClient is not active")
		payload = {"op": op, **kwargs}
		self._sock.sendall(encode_frame(payload))
		return read_frame(self._sock)

	def __exit__(self, exc_type, exc, tb) -> None:
		try:
			if self._sock is not None and exc_type is None:
				try:
					self.call("shutdown")
				except (OSError, ConnectionError):
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

	def _kill_process(self) -> None:
		if self._process is None:
			return
		self._process.terminate()
		try:
			self._process.wait(timeout=1.0)
		except subprocess.TimeoutExpired:
			self._process.kill()
			self._process.wait()
