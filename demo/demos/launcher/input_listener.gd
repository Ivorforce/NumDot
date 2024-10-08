extends Node

func _input(event: InputEvent) -> void:
	if event is InputEventKey:
		if event.keycode == KEY_B:
			TransitionManager.change_scene("res://demos/launcher/launcher.tscn")
	
