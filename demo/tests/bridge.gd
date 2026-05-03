extends Node

const CONNECT_TIMEOUT_SEC := 2.0

var peer: StreamPeerTCP


func _ready() -> void:
	if not OS.has_environment("NUMDOT_BRIDGE_PORT"):
		printerr("bridge: NUMDOT_BRIDGE_PORT is not set")
		get_tree().quit(1)
		return

	var port := OS.get_environment("NUMDOT_BRIDGE_PORT").to_int()
	if port <= 0 or port > 65535:
		printerr("bridge: invalid NUMDOT_BRIDGE_PORT: ", OS.get_environment("NUMDOT_BRIDGE_PORT"))
		get_tree().quit(1)
		return

	peer = StreamPeerTCP.new()
	var err := peer.connect_to_host("127.0.0.1", port)
	if err != OK:
		printerr("bridge: connect_to_host failed: ", err)
		get_tree().quit(1)
		return

	var deadline := Time.get_ticks_msec() + int(CONNECT_TIMEOUT_SEC * 1000.0)
	while true:
		peer.poll()
		var status := peer.get_status()
		if status == StreamPeerTCP.STATUS_CONNECTED:
			break
		if status == StreamPeerTCP.STATUS_ERROR or status == StreamPeerTCP.STATUS_NONE:
			printerr("bridge: connection failed, status=", status)
			get_tree().quit(1)
			return
		if Time.get_ticks_msec() > deadline:
			printerr("bridge: connect timed out")
			get_tree().quit(1)
			return
		await get_tree().process_frame

	peer.set_no_delay(true)
	print("bridge: connected to 127.0.0.1:", port)

	await _run_loop()


func _run_loop() -> void:
	while true:
		peer.poll()
		if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			printerr("bridge: peer disconnected")
			get_tree().quit(1)
			return

		# Read 4-byte length prefix.
		var header := await _read_exact(4)
		if header.is_empty():
			return  # quit() already called by _read_exact on failure
		var payload_len := header.decode_u32(0)

		# Read payload.
		var payload_bytes := await _read_exact(payload_len)
		if payload_bytes.is_empty() and payload_len != 0:
			return

		var payload_str := payload_bytes.get_string_from_utf8()
		var request: Variant = JSON.parse_string(payload_str)
		if typeof(request) != TYPE_DICTIONARY:
			_send({"ok": false, "error": "invalid_json", "raw": payload_str})
			continue

		var op: String = request.get("op", "")
		var response := _dispatch(op, request)
		_send(response)

		if op == "shutdown":
			get_tree().quit(0)
			return


func _dispatch(op: String, _request: Dictionary) -> Dictionary:
	match op:
		"ping":
			return {"ok": true, "pong": true}
		"shutdown":
			return {"ok": true}
		_:
			return {"ok": false, "error": "unknown_op", "op": op}


func _read_exact(n: int) -> PackedByteArray:
	var buf := PackedByteArray()
	while buf.size() < n:
		peer.poll()
		if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			printerr("bridge: peer disconnected mid-read")
			get_tree().quit(1)
			return PackedByteArray()
		var available := peer.get_available_bytes()
		if available <= 0:
			await get_tree().process_frame
			continue
		var to_read := mini(available, n - buf.size())
		var result: Array = peer.get_data(to_read)
		var err: int = result[0]
		if err != OK:
			printerr("bridge: get_data failed: ", err)
			get_tree().quit(1)
			return PackedByteArray()
		buf.append_array(result[1])
	return buf


func _send(payload: Dictionary) -> void:
	var json_str := JSON.stringify(payload)
	var json_bytes := json_str.to_utf8_buffer()
	var frame := PackedByteArray()
	frame.resize(4)
	frame.encode_u32(0, json_bytes.size())
	frame.append_array(json_bytes)
	var err := peer.put_data(frame)
	if err != OK:
		printerr("bridge: put_data failed: ", err)
		get_tree().quit(1)
