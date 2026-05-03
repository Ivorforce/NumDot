def test_ping(bridge):
	assert bridge.call("ping") == {"ok": True, "pong": True}


def test_unknown_op(bridge):
	resp = bridge.call("definitely_not_a_real_op")
	assert resp["ok"] is False
	assert resp["error"] == "unknown_op"
	assert resp["op"] == "definitely_not_a_real_op"


def test_multiple_pings(bridge):
	for _ in range(5):
		assert bridge.call("ping") == {"ok": True, "pong": True}
