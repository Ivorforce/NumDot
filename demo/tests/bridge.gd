extends Node

const CONNECT_TIMEOUT_SEC := 2.0

const ND_DTYPES := {
	"Int8": nd.Int8,
	"Int16": nd.Int16,
	"Int32": nd.Int32,
	"Int64": nd.Int64,
	"UInt8": nd.UInt8,
	"UInt16": nd.UInt16,
	"UInt32": nd.UInt32,
	"UInt64": nd.UInt64,
	"Float32": nd.Float32,
	"Float64": nd.Float64,
	"Bool": nd.Bool,
	"Complex64": nd.Complex64,
	"Complex128": nd.Complex128,
}

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
		"nd_call":
			return _op_nd_call(request, blobs)
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


func _op_nd_call(request: Dictionary, blobs: Array) -> Dictionary:
	var func_name: String = request.get("func", "")
	if func_name == "":
		return _nd_error(func_name, "missing 'func'")

	var arg_specs: Array = request.get("args", [])
	var decoded_args: Array = []
	for i in arg_specs.size():
		var decoded := _decode_arg(arg_specs[i], blobs)
		if not decoded.get("ok", false):
			return _nd_error(func_name, "arg %d: %s" % [i, decoded.get("error", "decode failed")])
		decoded_args.append(decoded["value"])

	if not ClassDB.class_has_method("nd", func_name):
		return _nd_error(func_name, "unknown function")

	var call_args: Array = ["nd", func_name]
	call_args.append_array(decoded_args)
	var result = Callable(ClassDB, "class_call_static").callv(call_args)
	if result == null:
		return _nd_error(func_name, "function returned null")

	var dumped = nd.dumpb(result)
	if dumped == null:
		return _nd_error(func_name, "nd.dumpb returned null")
	return {"header": {"ok": true}, "blobs": [dumped]}


func _decode_arg(spec: Variant, blobs: Array) -> Dictionary:
	if typeof(spec) != TYPE_DICTIONARY:
		return {"ok": false, "error": "arg spec must be a dict, got %s" % typeof(spec)}
	if spec.has("$blob"):
		var idx: int = spec["$blob"]
		if idx < 0 or idx >= blobs.size():
			return {"ok": false, "error": "blob index %d out of range" % idx}
		var arr = nd.load(blobs[idx])
		if arr == null:
			return {"ok": false, "error": "nd.load failed for blob %d" % idx}
		return {"ok": true, "value": arr}
	if spec.has("$blobs"):
		# Homogeneous list of ndarrays — one .npy blob per element. Decode into
		# a GDScript Array of NDArray, which is what variant_to_vector consumes
		# (e.g. nd.concatenate's first arg).
		var indices: Array = spec["$blobs"]
		var out: Array = []
		for v in indices:
			var idx: int = v
			if idx < 0 or idx >= blobs.size():
				return {"ok": false, "error": "$blobs index %d out of range" % idx}
			var loaded = nd.load(blobs[idx])
			if loaded == null:
				return {"ok": false, "error": "nd.load failed for $blobs[%d]" % idx}
			out.append(loaded)
		return {"ok": true, "value": out}
	if spec.has("$value"):
		return {"ok": true, "value": spec["$value"]}
	if spec.has("$int"):
		return {"ok": true, "value": int(spec["$int"])}
	if spec.has("$float"):
		# 8-byte raw blob carrying a bit-exact IEEE 754 double. Decoded as a
		# Variant FLOAT so NumDot's binding sees it as a Python scalar (and
		# applies weak-scalar promotion), not a typed 0-d ndarray.
		var idx: int = spec["$float"]
		if idx < 0 or idx >= blobs.size():
			return {"ok": false, "error": "float blob index %d out of range" % idx}
		var blob: PackedByteArray = blobs[idx]
		if blob.size() != 8:
			return {"ok": false, "error": "float blob must be 8 bytes, got %d" % blob.size()}
		return {"ok": true, "value": blob.decode_double(0)}
	if spec.has("$ints"):
		var arr: Array = []
		for v in spec["$ints"]:
			arr.append(int(v))
		return {"ok": true, "value": arr}
	if spec.has("$dtype"):
		var name: String = spec["$dtype"]
		if not ND_DTYPES.has(name):
			return {"ok": false, "error": "unknown dtype %s" % name}
		return {"ok": true, "value": ND_DTYPES[name]}
	if spec.has("$null"):
		return {"ok": true, "value": null}
	return {"ok": false, "error": "arg spec has no recognised tag"}


func _nd_error(func_name: String, message: String) -> Dictionary:
	return {"header": {"ok": false, "error": "op_failed", "op": "nd_call", "func": func_name, "message": message}, "blobs": []}


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
