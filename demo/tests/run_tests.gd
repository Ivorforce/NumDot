extends Node

func _ready():
	var script = GDScript.new()
	script.set_source_code(FileAccess.open("res://tests/gen/tests.gd", FileAccess.READ).get_as_text())
	script.reload()
	var ref = RefCounted.new()
	ref.set_script(script)
	return ref.run()
