extends Node2D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var test_size := 2000000
	var test_count := 1
		
	var a_packed := PackedInt32Array()
	a_packed.resize(test_size)
	a_packed.fill(1)
	var a_nd = nd.ones(test_size, nd.DType.Int32)

	## Test 1: Create packed arrays
	var start_time := Time.get_ticks_usec()
	for t in test_count:
		var b_packed := PackedInt32Array()
		b_packed.resize(test_size)
		b_packed.fill(1)
	print(Time.get_ticks_usec() - start_time)

	# Test 2: Create nd arrays
	start_time = Time.get_ticks_usec()
	for t in test_count:
		var b_nd = nd.ones(test_size, nd.DType.Int32)
	print(Time.get_ticks_usec() - start_time)

	# Test 3: Multiply packed arrays
	start_time = Time.get_ticks_usec()
	for t in test_count:
		var b_packed := PackedInt32Array()
		b_packed.resize(test_size)
		for i in test_size:
			b_packed[i] = a_packed[i] * a_packed[i]
	print(Time.get_ticks_usec() - start_time)
	
	# Test 4: Multiply nd arrays
	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.multiply(a_nd, a_nd)
	print(Time.get_ticks_usec() - start_time)
	
	# Test 5: sin packed arrays
	start_time = Time.get_ticks_usec()
	for t in test_count:
		var b_packed := PackedInt32Array()
		b_packed.resize(test_size)
		for i in test_size:
			b_packed[i] = sin(a_packed[i])
	print(Time.get_ticks_usec() - start_time)
	
	# Test 6: sin nd arrays
	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.sin(a_nd)
	print(Time.get_ticks_usec() - start_time)
	
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
