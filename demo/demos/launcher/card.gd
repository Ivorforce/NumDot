extends Control

var scene_path: String

func set_data(data: Dictionary) -> void:
	%Name.text = "  " + data["name"] + "  "
	%Description.text = data["description"]
	%Link.uri = data["link"]
	scene_path = data["path"] + "main.tscn"

func _on_name_pressed() -> void:
	TransitionManager.change_scene(scene_path)
