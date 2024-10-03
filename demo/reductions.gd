extends Benchmark

func run_gdscript(
	test_size: int,
	test_count: int,
):
	var a_packed := PackedFloat32Array()
	a_packed.resize(test_size)
	var b_packed := PackedByteArray()
	b_packed.resize(test_size)
	
	begin_section("sum")
	for t in test_count:
		var sum := 0.0
		for i in test_size:
			sum += a_packed[i]
	store_result()

	begin_section("std")
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
	store_result()

	begin_section("all")
	for t in test_count:
		var all := true
		for i in test_size:
			all = all && b_packed[i]
	store_result()

func run_numdot_nd(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.ones(test_size, nd.DType.Bool)

	begin_section("sum")
	for t in test_count:
		nd.sum(a_nd)
	store_result()
	
	begin_section("std")
	for t in test_count:
		nd.std(a_nd)
	store_result()

	begin_section("all")
	for t in test_count:
		nd.all(b_nd)
	store_result()

func run_numdot_ndifb(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.ones(test_size, nd.DType.Bool)

	begin_section("sum")
	for t in test_count:
		ndi.sum(a_nd)
	store_result()
	
	begin_section("std")
	for t in test_count:
		ndf.std(a_nd)
	store_result()

	begin_section("all")
	for t in test_count:
		ndb.all(b_nd)
	store_result()


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

	end()
