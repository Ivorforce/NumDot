extends Node2D

func _ready() -> void:
	# Edit your testing code here
	# But don't submit the changes to the repository!
	pass
	var f = FileAccess.open("/Users/lukas/test.npy", FileAccess.READ)
	print(nd.load(f))
