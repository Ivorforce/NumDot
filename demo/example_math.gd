extends Node2D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var test_size := 50000000
	
	var start_time := Time.get_ticks_msec()
	var a_packed := PackedVector2Array()
	for i in test_size:
		a_packed.append(Vector2(1, 1))
	print(Time.get_ticks_msec() - start_time)

	start_time = Time.get_ticks_msec()
	var a_nd = ND.ones([test_size, 2])
	print(Time.get_ticks_msec() - start_time)

	start_time = Time.get_ticks_msec()
	for i in test_size:
		a_packed[i] *= a_packed[i]
	print(Time.get_ticks_msec() - start_time)
	
	start_time = Time.get_ticks_msec()
	a_nd = ND.multiply(a_nd, a_nd)
	print(Time.get_ticks_msec() - start_time)
	
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
