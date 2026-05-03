import json
import socket
import struct


def encode_frame(payload: dict) -> bytes:
	body = json.dumps(payload).encode("utf-8")
	return struct.pack("<I", len(body)) + body


def _recv_exact(sock: socket.socket, n: int) -> bytes:
	chunks = []
	remaining = n
	while remaining > 0:
		chunk = sock.recv(remaining)
		if not chunk:
			raise ConnectionError(f"peer closed connection with {remaining} bytes still expected")
		chunks.append(chunk)
		remaining -= len(chunk)
	return b"".join(chunks)


def read_frame(sock: socket.socket) -> dict:
	header = _recv_exact(sock, 4)
	(length,) = struct.unpack("<I", header)
	body = _recv_exact(sock, length) if length else b""
	return json.loads(body.decode("utf-8"))
