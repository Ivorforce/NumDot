def test_ping(bridge):
	header, blobs = bridge.call("ping")
	assert header == {"ok": True, "pong": True}
	assert blobs == []


def test_unknown_op(bridge):
	header, blobs = bridge.call("definitely_not_a_real_op")
	assert header["ok"] is False
	assert header["error"] == "unknown_op"
	assert header["op"] == "definitely_not_a_real_op"
	assert blobs == []


def test_multiple_pings(bridge):
	for _ in range(5):
		header, blobs = bridge.call("ping")
		assert header == {"ok": True, "pong": True}
		assert blobs == []
