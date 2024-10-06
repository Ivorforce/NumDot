extends Benchmark

func run_gdscript(
	test_size: int,
	test_count: int,
):
	var a_packed := PackedFloat32Array()
	a_packed.resize(test_size)
	
	begin_section("tan")
	for t in test_count:
		for i in test_size:
			a_packed[i] = tan(a_packed[i])
	store_result()

	begin_section("sin")
	for t in test_count:
		for i in test_size:
			a_packed[i] = sin(a_packed[i])
	store_result()

	begin_section("asinh")
	for t in test_count:
		for i in test_size:
			a_packed[i] = asinh(a_packed[i])
	store_result()

func run_numdot_nd(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)

	begin_section("tan")
	for t in test_count:
		nd.tan(a_nd)
	store_result()
	
	begin_section("sin")
	for t in test_count:
		nd.sin(a_nd)
	store_result()

	begin_section("asinh")
	for t in test_count:
		nd.asinh(a_nd)
	store_result()


func run_numdot_inplace(
	test_size: int,
	test_count: int,
):
	var a_nd := nd.ones(test_size, nd.DType.Float32)

	begin_section("tan")
	for t in test_count:
		a_nd.assign_tan(a_nd)
	store_result()

	begin_section("sin")
	for t in test_count:
		a_nd.assign_sin(a_nd)
	store_result()

	begin_section("asinh")
	for t in test_count:
		a_nd.assign_asinh(a_nd)
	store_result()


func run_benchmark():
	const test_size := 50000
	const test_count := 100

	print("Trigonometry with size=%d count: %d" % [test_size, test_count])

	print("GDScript:")
	run_gdscript(test_size, test_count)

	print("NumDot nd:")
	run_numdot_nd(test_size, test_count)

	print("NumDot inplace:")
	run_numdot_inplace(test_size, test_count)

	end()
