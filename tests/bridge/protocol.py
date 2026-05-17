import json
import socket
import struct
from typing import Sequence


def encode_frame(header: dict, blobs: Sequence[bytes] = ()) -> bytes:
	# Only emit "blobs" sizes when there are blobs, so zero-blob frames
	# stay byte-identical to the v0 protocol.
	if blobs:
		header = {**header, "blobs": [len(b) for b in blobs]}
	body = json.dumps(header).encode("utf-8")
	frame = struct.pack("<I", len(body)) + body
	for blob in blobs:
		frame += blob
	return frame


def _recv_exact(sock: socket.socket, n: int) -> bytes:
	if n == 0:
		return b""
	chunks = []
	remaining = n
	while remaining > 0:
		chunk = sock.recv(remaining)
		if not chunk:
			raise ConnectionError(f"peer closed connection with {remaining} bytes still expected")
		chunks.append(chunk)
		remaining -= len(chunk)
	return b"".join(chunks)


def read_frame(sock: socket.socket) -> tuple[dict, list[bytes]]:
	length_bytes = _recv_exact(sock, 4)
	(header_len,) = struct.unpack("<I", length_bytes)
	header_bytes = _recv_exact(sock, header_len) if header_len else b""
	header = json.loads(header_bytes.decode("utf-8"))
	blob_sizes = header.get("blobs", [])
	blobs = [_recv_exact(sock, int(size)) for size in blob_sizes]
	return header, blobs
