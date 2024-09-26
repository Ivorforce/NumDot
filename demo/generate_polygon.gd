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
	#  because it could immediately return after the math, rather than needing
	#  another copy.
	var polygon := nd.empty([num_sides, 2])
	
	polygon.get(null, 0).assign_sin(angle_delta)
	polygon.get(null, 0).assign_cos(angle_delta)
	polygon.assign_multiply(polygon, radius)
	polygon.assign_add(polygon, position)

	# In this test we count conversion to the packed array
	#  as part of the test, because that's what we'd need to pass to godot.
	return polygon.to_packed_vector2_array()

func _ready():
	const RADIUS := 1.0
	const NUM_SIDES := 20000000
	const POSITION := Vector2(0.0, 0.0)

	# Examples from my computer:
	# n=200            26 GDScript
	#                  92 NumDot
	# n=2000          115 GDScript
	#                 144 NumDot
	# n=20000         789 GDScript
	#                 910 NumDot
	# n=200000       8014 GDScript
	#                7116 NumDot
	# n=2000000     84384 GDScript
	#               91861 NumDot
	# n=20000000   804691 GDScript
	#              795497 NumDot

	var start_time: int
	print("Generate polygon with sides=" + str(NUM_SIDES))

	start_time = Time.get_ticks_usec()
	var result_gd := run_gdscript(RADIUS, NUM_SIDES, POSITION)
	print("GDScript: " + str(Time.get_ticks_usec() - start_time))

	start_time = Time.get_ticks_usec()
	var result_nd := run_numdot(RADIUS, NUM_SIDES, POSITION)
	print("NumDot: " + str(Time.get_ticks_usec() - start_time))
