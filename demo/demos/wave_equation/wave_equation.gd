extends Node2D

@export var num_points: int = 3000

@export_category("Simulation parameters")
@export var xmin: float = 0.
@export var xmax: float = 1.
@export var wave_speed: float = 0.25
@export var frame_rate: float = 60

@export_category("Visualization parameters")
@export var xscale: float = 500.
@export var yscale: float = 50.
@export var num_draw_points: int = 200

@export_category("Solver parameters")
@export var solver: Solver

# simulation parameters computed
@onready var dx: float = (xmax - xmin)/num_points
@onready var num_steps_per_frame: int = max(1, ceili(wave_speed/ dx / frame_rate)) # to satisfy CFL condition
	
func _ready() -> void:
	solver.initialize()
	
func _process(delta: float) -> void:
	%FPSLabel.text = "FPS: " + str(int(1/delta))
	%IterLabel.text = "Sub-steps: " + str(num_steps_per_frame)

	solver.simulation_step(delta)
	queue_redraw()

func _draw() -> void:
	solver.on_draw()

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	solver.initialize()

func _on_point_slider_drag_ended(value_changed: bool) -> void:
	%PointLabel.text = "Points: " + str($PointSlider.value)
	%CFLLabel.text = "CFL: " + str(snappedf(wave_speed * 1/(frame_rate * num_steps_per_frame * dx), 1e-3))
	
	num_points = $PointSlider.value
	dx = (xmax - xmin)/num_points
	num_steps_per_frame = max(1, ceili(wave_speed/ dx / frame_rate)) # to satisfy CFL condition
	solver.initialize()
