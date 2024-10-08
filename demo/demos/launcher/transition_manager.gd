extends CanvasLayer

func _ready() -> void:
	layer = 2
	
func change_scene(target: String) -> void:
	var tween = get_tree().create_tween()
	tween.tween_property($ColorRect, "modulate:a", 1., 0.75)
	tween.tween_callback(func(): get_tree().change_scene_to_file(target))
	tween.tween_property($ColorRect, "modulate:a", 0., 0.75)	
