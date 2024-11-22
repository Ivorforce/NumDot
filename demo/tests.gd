extends Node2D

func _ready() -> void:
	# Edit your testing code here
	# But don't submit the changes to the repository!
	print(nd.full([100], 2, nd.Bool).to_packed_byte_array())
