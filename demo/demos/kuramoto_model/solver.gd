extends Node
class_name KuramotoSolver

@export var params: Node2D
var integrator: Array[Callable]

func initialize() -> void:
	pass

func simulation_step() -> void:
	pass
	
func on_draw() -> void:
	pass

func generate_frequencies() -> void:
	pass
	
func set_integrator(idx: int) -> void:
	pass
