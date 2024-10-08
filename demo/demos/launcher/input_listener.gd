extends Node

func _input(event: InputEvent) -> void:
	if event is InputEventKey:
		if event.is_action_pressed("back"):
			TransitionManager.change_scene("res://demos/launcher/launcher.tscn")
	
