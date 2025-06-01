extends Node2D

var demo_directory := "res://demos"
var demo_data = []
@onready var demo_list = %DemoList

var card_scene := preload("res://demos/launcher/card.tscn")

func _ready() -> void:
	retrieve_metadata()
	for demo in demo_data:
		var card := card_scene.instantiate()
		card.set_data(demo)
		demo_list.add_child(card)
		demo_list.add_spacer(false)
		
func _process(delta: float) -> void:
	pass

func retrieve_metadata():
	var dir = DirAccess.open(demo_directory)
	for demo in dir.get_directories():
		read_json("/".join([demo_directory, demo, "metadata.json"]))

func read_json(path: String) -> void:
	var file = FileAccess.open(path, FileAccess.READ)
	if file:
		var data = JSON.new().parse_string(file.get_as_text())
		if data.has("name") and data.has("description") and data.has("link"):
			data["path"] = path.get_base_dir() + "/"
			if ResourceLoader.exists(data["path"] + "main.tscn"):
				demo_data.append(data)
			else:
				print(data["path"] + " missing main.tscn!")
		else:
			print("JSON file (" + path + ") is missing required keys (name, description, link).")
		
func _on_texture_button_pressed() -> void:
	OS.shell_open("https://godotengine.org/asset-library/asset/3351")
