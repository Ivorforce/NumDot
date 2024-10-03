extends Node

func run_gdscript(
	test_size: int,
	test_count: int,
):
	var start_time: int
	var a_packed := PackedFloat32Array()
	a_packed.resize(test_size)
	
	start_time = Time.get_ticks_usec()
	for t in test_count:
		for i in test_size:
			var acc := a_packed[i]
			a_packed[i] = acc * acc
	print("mul: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		for i in test_size:
			a_packed[i] = sin(a_packed[i])
	print("sin: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		for i in test_size:
			a_packed[i] = asinh(a_packed[i])
	print("asinh: " + str(Time.get_ticks_usec() - start_time))

func run_numdot_nd(
	test_size: int,
	test_count: int,
):
	var start_time: int
	var a_nd := nd.ones(test_size, nd.DType.Float32)

	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.multiply(a_nd, a_nd)
	print("mul: " + str(Time.get_ticks_usec() - start_time))
	
	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.sin(a_nd)
	print("sin: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.asinh(a_nd)
	print("asinh: " + str(Time.get_ticks_usec() - start_time))


func run_numdot_inplace(
	test_size: int,
	test_count: int,
):
	var start_time: int
	var a_nd := nd.ones(test_size, nd.DType.Float32)

	start_time = Time.get_ticks_usec()
	for t in test_count:
		a_nd.assign_multiply(a_nd, a_nd)
	print("mul: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		a_nd.assign_sin(a_nd)
	print("sin: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		a_nd.assign_asinh(a_nd)
	print("asinh: " + str(Time.get_ticks_usec() - start_time))


func run_benchmark():
	const test_size := 50000
	const test_count := 100

	print("Math with size=%d count: %d" % [test_size, test_count])

	print("GDScript:")
	run_gdscript(test_size, test_count)

	print("NumDot nd:")
	run_numdot_nd(test_size, test_count)

	print("NumDot inplace:")
	run_numdot_inplace(test_size, test_count)

	print()
