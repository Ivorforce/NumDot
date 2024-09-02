extends Node2D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var test_size := 20000
	var test_count := 500
		
	
	# Test 1: Create packed arrays
	var start_time := Time.get_ticks_usec()
	for t in test_count:
		var a_packed := PackedInt32Array()
		for i in test_size:
			a_packed.append(1)
	print(Time.get_ticks_usec() - start_time)

	# Test 2: Create nd arrays
	start_time = Time.get_ticks_usec()
	for t in test_count:
		var a_nd = nd.ones(test_size, NDArray.DType.Int32)
	print(Time.get_ticks_usec() - start_time)

	# Test 3: Multiply packed arrays
	var a_packed := PackedInt32Array()
	for i in test_size:
		a_packed.append(1)
	
	start_time = Time.get_ticks_usec()
	for t in test_count:
		var b_packed := PackedInt32Array()
		for i in test_size:
			b_packed.append(a_packed[i] * a_packed[i])
	print(Time.get_ticks_usec() - start_time)
	
	# Test 4: Multiply nd arrays
	var a_nd = nd.ones(test_size, NDArray.DType.Int32)

	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.multiply(a_nd, a_nd)
	print(Time.get_ticks_usec() - start_time)
	
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
