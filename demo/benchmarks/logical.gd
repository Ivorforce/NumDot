extends Benchmark

func run_gdscript(
	test_size: int,
	test_count: int,
):
	var a_packed := PackedFloat32Array()
	a_packed.resize(test_size)
	var b_packed := PackedByteArray()
	
	begin_section("equal")
	for t in test_count:
		for i in test_size:
			var acc := a_packed[i]
			b_packed[i] = acc == acc
	store_result()

	begin_section("greater")
	for t in test_count:
		for i in test_size:
			var acc := a_packed[i]
			b_packed[i] = acc > acc
	store_result()

	begin_section("and")
	for t in test_count:
		for i in test_size:
			var acc := b_packed[i]
			b_packed[i] = acc and acc
	store_result()

func run_numdot_nd(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.ones(test_size, nd.DType.Bool)

	begin_section("equal")
	for t in test_count:
		nd.equal(a_nd, a_nd)
	store_result()
	
	begin_section("greater")
	for t in test_count:
		nd.greater(a_nd, a_nd)
	store_result()

	begin_section("and")
	for t in test_count:
		nd.logical_and(b_nd, b_nd)
	store_result()


func run_numdot_nd_scalar(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.ones(test_size, nd.DType.Bool)
	var scalar = nd.float32(5)
	var scalar_bool = nd.bool_(false)

	begin_section("equal")
	for t in test_count:
		nd.equal(a_nd, scalar)
	store_result()
	
	begin_section("greater")
	for t in test_count:
		nd.greater(a_nd, scalar)
	store_result()

	begin_section("and")
	for t in test_count:
		nd.logical_and(b_nd, scalar_bool)
	store_result()

func run_numdot_inplace(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.ones(test_size, nd.DType.Bool)

	begin_section("equal")
	for t in test_count:
		b_nd.assign_equal(a_nd, a_nd)
	store_result()

	begin_section("greater")
	for t in test_count:
		b_nd.assign_greater(a_nd, a_nd)
	store_result()

	begin_section("and")
	for t in test_count:
		b_nd.assign_logical_and(b_nd, b_nd)
	store_result()


func run_numdot_inplace_scalar(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.ones(test_size, nd.DType.Bool)
	var scalar = nd.float32(5)
	var scalar_bool = nd.bool_(false)

	begin_section("equal")
	for t in test_count:
		b_nd.assign_equal(a_nd, scalar)
	store_result()

	begin_section("greater")
	for t in test_count:
		b_nd.assign_greater(a_nd, scalar)
	store_result()

	begin_section("and")
	for t in test_count:
		b_nd.assign_logical_and(b_nd, scalar_bool)
	store_result()


func run_benchmark():
	const test_size := 50000
	const test_count := 100

	print("Math with size=%d count: %d" % [test_size, test_count])

	print("GDScript:")
	run_gdscript(test_size, test_count)

	print("NumDot nd:")
	run_numdot_nd(test_size, test_count)

	print("NumDot nd scalar:")
	run_numdot_nd_scalar(test_size, test_count)

	print("NumDot inplace:")
	run_numdot_inplace(test_size, test_count)

	print("NumDot inplace scalar:")
	run_numdot_inplace_scalar(test_size, test_count)

	end()
