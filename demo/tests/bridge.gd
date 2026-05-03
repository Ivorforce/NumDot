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

		# Read 4-byte header length prefix.
		var length_bytes := await _read_exact(4)
		if length_bytes.is_empty():
			return  # quit() already called by _read_exact on failure
		var header_len := length_bytes.decode_u32(0)

		# Read JSON header.
		var header_bytes := await _read_exact(header_len)
		if header_bytes.is_empty() and header_len != 0:
			return

		var header_str := header_bytes.get_string_from_utf8()
		var request: Variant = JSON.parse_string(header_str)
		if typeof(request) != TYPE_DICTIONARY:
			_send({"ok": false, "error": "invalid_json", "raw": header_str}, [])
			continue

		# Read blob payloads, if any.
		var blob_sizes: Array = request.get("blobs", [])
		var blobs: Array = []
		var read_failed := false
		for size_v in blob_sizes:
			var size := int(size_v)
			var blob := await _read_exact(size)
			if blob.is_empty() and size != 0:
				read_failed = true
				break
			blobs.append(blob)
		if read_failed:
			return

		var op: String = request.get("op", "")
		var result := _dispatch(op, request, blobs)
		_send(result["header"], result["blobs"])

		if op == "shutdown":
			get_tree().quit(0)
			return


func _dispatch(op: String, request: Dictionary, blobs: Array) -> Dictionary:
	match op:
		"ping":
			return {"header": {"ok": true, "pong": true}, "blobs": []}
		"shutdown":
			return {"header": {"ok": true}, "blobs": []}
		"echo_blob":
			return {"header": {"ok": true}, "blobs": blobs}
		"echo_array":
			return _op_echo_array(blobs)
		"add":
			return _op_add(blobs)
		_:
			return {"header": {"ok": false, "error": "unknown_op", "op": op}, "blobs": []}


func _op_echo_array(blobs: Array) -> Dictionary:
	var out: Array = []
	for blob in blobs:
		var arr = nd.load(blob)
		if arr == null:
			return _op_error("echo_array", "nd.load returned null")
		var dumped = nd.dumpb(arr)
		if dumped == null:
			return _op_error("echo_array", "nd.dumpb returned null")
		out.append(dumped)
	return {"header": {"ok": true}, "blobs": out}


func _op_add(blobs: Array) -> Dictionary:
	if blobs.size() != 2:
		return _op_error("add", "expected 2 blobs, got %d" % blobs.size())
	var a = nd.load(blobs[0])
	if a == null:
		return _op_error("add", "nd.load(blobs[0]) returned null")
	var b = nd.load(blobs[1])
	if b == null:
		return _op_error("add", "nd.load(blobs[1]) returned null")
	var result = nd.add(a, b)
	if result == null:
		return _op_error("add", "nd.add returned null")
	var dumped = nd.dumpb(result)
	if dumped == null:
		return _op_error("add", "nd.dumpb returned null")
	return {"header": {"ok": true}, "blobs": [dumped]}


func _op_error(op: String, message: String) -> Dictionary:
	return {"header": {"ok": false, "error": "op_failed", "op": op, "message": message}, "blobs": []}


func _read_exact(n: int) -> PackedByteArray:
	if n == 0:
		return PackedByteArray()
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


func _send(header: Dictionary, blobs: Array) -> void:
	# Only emit the "blobs" field when there are blobs to attach, so
	# zero-blob frames remain byte-identical to the v0 protocol.
	if not blobs.is_empty():
		var sizes: Array = []
		for blob in blobs:
			sizes.append((blob as PackedByteArray).size())
		header["blobs"] = sizes

	var json_str := JSON.stringify(header)
	var json_bytes := json_str.to_utf8_buffer()
	var frame := PackedByteArray()
	frame.resize(4)
	frame.encode_u32(0, json_bytes.size())
	frame.append_array(json_bytes)
	for blob in blobs:
		frame.append_array(blob)

	var err := peer.put_data(frame)
	if err != OK:
		printerr("bridge: put_data failed: ", err)
		get_tree().quit(1)
