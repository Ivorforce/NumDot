extends Node

func run_gdscript(
	test_size: int,
	test_count: int,
):
	var start_time: int
	var a_packed := PackedFloat32Array()
	a_packed.resize(test_size)
	var b_packed := PackedByteArray()
	b_packed.resize(test_size)
	
	start_time = Time.get_ticks_usec()
	for t in test_count:
		var sum := 0.0
		for i in test_size:
			sum += a_packed[i]
	print("sum: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		var mean := 0.0
		for i in test_size:
			mean += a_packed[i]
		mean /= test_size
		var std := 0.0
		for i in test_size:
			var acc := a_packed[i] - mean
			std += acc * acc
		std = sqrt(std / (test_size - 1))
	print("std: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		var all := true
		for i in test_size:
			all = all && b_packed[i]
	print("all: " + str(Time.get_ticks_usec() - start_time))

func run_numdot_nd(
	test_size: int,
	test_count: int,
):
	var start_time: int
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.ones(test_size, nd.DType.Bool)

	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.sum(a_nd)
	print("sum: " + str(Time.get_ticks_usec() - start_time))
	
	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.std(a_nd)
	print("std: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		nd.all(b_nd)
	print("all: " + str(Time.get_ticks_usec() - start_time))

func run_numdot_ndifb(
	test_size: int,
	test_count: int,
):
	var start_time: int
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.ones(test_size, nd.DType.Bool)

	start_time = Time.get_ticks_usec()
	for t in test_count:
		ndi.sum(a_nd)
	print("sum: " + str(Time.get_ticks_usec() - start_time))
	
	start_time = Time.get_ticks_usec()
	for t in test_count:
		ndf.std(a_nd)
	print("std: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	for t in test_count:
		ndb.all(b_nd)
	print("all: " + str(Time.get_ticks_usec() - start_time))


func run_benchmark():
	const test_size := 50000
	const test_count := 100

	print("Reductions with size=%d count: %d" % [test_size, test_count])

	print("GDScript:")
	run_gdscript(test_size, test_count)

	print("NumDot nd:")
	run_numdot_nd(test_size, test_count)

	print("NumDot ndi / ndf / ndb:")
	run_numdot_ndifb(test_size, test_count)

	print()
