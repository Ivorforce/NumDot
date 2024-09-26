extends Node
# loosely adapted the non-benchmark part of https://github.com/godotengine/godot-benchmarks/blob/main/benchmarks/math/triangulate.gd

func run_gdscript(
	radius: float,
	num_sides: int,
	position: Vector2
) -> PackedVector2Array:
	var angle_delta: float = (PI * 2) / num_sides
	var vector: Vector2 = Vector2(radius, 0)
	var polygon := PackedVector2Array()
	polygon.resize(num_sides)

	for i in num_sides:
		polygon[i] = vector + position
		vector = vector.rotated(angle_delta)

	return polygon

func run_numdot(
	radius: float,
	num_sides: int,
	position: Vector2
) -> PackedVector2Array:
	var angle_delta := nd.linspace(0, PI * 2, num_sides, nd.DType.Float32)
	
	# TODO Would be better as a PackedVector2Array backed NDArray
	# because it could immediately return after the math.
	var polygon := nd.empty([num_sides, 2])
	
	polygon.get(null, 0).assign_sin(angle_delta)
	polygon.get(null, 0).assign_cos(angle_delta)
	polygon.assign_multiply(polygon, radius)
	polygon.assign_add(polygon, position)

	# In this test we count conversion to the packed array
	#  as part of the test, because that's what we'd need to pass to godot.
	
	# TODO needs to_packed_vector2_array()
	var polygon_packed := PackedVector2Array()
	polygon_packed.resize(num_sides)
	for i in num_sides:
		polygon_packed[i] = Vector2(polygon.get_float(i, 0), polygon.get_float(i, 1))
	return polygon_packed

func _ready():
	const RADIUS := 1.0
	const NUM_SIDES := 7_0000
	const POSITION := Vector2(0.0, 0.0)
	
	# In this test, GDScript speed currently wins.
	# This is because of the two above mentioned ToDos.
	# With both implemented, the result should somewhat favor NumDot
	# (2600 gdscript vs 2500 NumDot)
	# It should be able to reach speeds even faster than this, but
	#  i don't know what the next bottleneck is.
	
	var start_time: int
	print("Generate polygon with sides=" + str(NUM_SIDES))

	start_time = Time.get_ticks_usec()
	var result_gd := run_gdscript(RADIUS, NUM_SIDES, POSITION)
	print("GDScript: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	var result_nd := run_numdot(RADIUS, NUM_SIDES, POSITION)
	print("NumDot: " + str(Time.get_ticks_usec() - start_time))
