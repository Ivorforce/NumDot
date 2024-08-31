extends Node2D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var test_size := 500000
	
	var a_packed := PackedFloat64Array()
	for i in test_size:
		a_packed.append(1)
	
	var a_nd = ND.ones(test_size)
	
	var start_time := Time.get_ticks_msec()
	for i in test_size:
		a_packed[i] += a_packed[i]
	print(Time.get_ticks_msec() - start_time)
	
	start_time = Time.get_ticks_msec()
	a_nd = ND.add(a_nd, a_nd)
	print(Time.get_ticks_msec() - start_time)
	
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
