extends Node2D

func _ready() -> void:
	# Edit your testing code here
	# But don't submit the changes to the repository!
	pass
	var packed = PackedFloat32Array([2, 3, 4])
	var a = nd.array(packed)
	packed[0] = 100
	a.set(500)
	print(a)
	print(packed)
