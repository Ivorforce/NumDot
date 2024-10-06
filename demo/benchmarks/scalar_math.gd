extends Benchmark

func run_gdscript(
	test_size: int,
	test_count: int,
):
	var a_packed := PackedFloat32Array()
	a_packed.resize(test_size)
	
	begin_section("add")
	for t in test_count:
		for i in test_size:
			a_packed[i] = a_packed[i] + 5
	store_result()

	begin_section("mul")
	for t in test_count:
		for i in test_size:
			a_packed[i] = a_packed[i] * 5
	store_result()

	begin_section("pow")
	for t in test_count:
		for i in test_size:
			a_packed[i] = pow(a_packed[i], 5)
	store_result()

func run_numdot_nd(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.float32(5)

	begin_section("add")
	for t in test_count:
		nd.add(a_nd, b_nd)
	store_result()
	
	begin_section("mul")
	for t in test_count:
		nd.multiply(a_nd, b_nd)
	store_result()

	begin_section("pow")
	for t in test_count:
		nd.pow(a_nd, b_nd)
	store_result()


func run_numdot_inplace(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)
	var b_nd := nd.float32(5)

	begin_section("add")
	for t in test_count:
		a_nd.assign_add(a_nd, b_nd)
	store_result()

	begin_section("mul")
	for t in test_count:
		a_nd.assign_multiply(a_nd, b_nd)
	store_result()

	begin_section("pow")
	for t in test_count:
		a_nd.assign_pow(a_nd, b_nd)
	store_result()


func run_benchmark():
	const test_size := 50000
	const test_count := 100

	print("Scalar Math with size=%d count: %d" % [test_size, test_count])

	print("GDScript:")
	run_gdscript(test_size, test_count)

	print("NumDot nd:")
	run_numdot_nd(test_size, test_count)

	print("NumDot inplace:")
	run_numdot_inplace(test_size, test_count)

	end()
