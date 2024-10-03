extends Benchmark

func run_gdscript(
	test_size: int,
	test_count: int,
):	
	var projection := Projection()
	var vector = Vector4(0, 1, 2, 3)
	
	var matrix_flat := PackedFloat32Array()
	matrix_flat.resize(test_size * 4)
	
	begin_section("projection")
	for t in test_count:
		projection * vector
	store_result()

	begin_section("matrix")
	for t in test_count:
		# TODO No idea if this is correct, but it should be the correct number of operations.
		var vector_result := Vector4()
		for x in test_size:
			for y in 4:
				vector_result[y] += matrix_flat[x * 4 + y]
	store_result()

func run_numdot(
	test_size: int,
	test_count: int,
):
	var projection := nd.ones([4, 4], nd.DType.Float32)
	var vector := nd.arange(0, 4, 1, nd.DType.Float32)
	
	var matrix := nd.ones([test_size, 4], nd.DType.Float32)

	begin_section("projection")
	for t in test_count:
		nd.matmul(projection, vector.get(nd.newaxis(), null))  # TODO This should be supported without the get
	store_result()
	
	begin_section("matrix")
	for t in test_count:
		nd.matmul(matrix, vector.get(nd.newaxis(), null))
	store_result()

func run_benchmark():
	const test_size := 1000
	const test_count := 100

	print("Matrix with rows=%d count: %d" % [test_size, test_count])

	print("GDScript:")
	run_gdscript(test_size, test_count)

	print("NumDot")
	run_numdot(test_size, test_count)

	end()
