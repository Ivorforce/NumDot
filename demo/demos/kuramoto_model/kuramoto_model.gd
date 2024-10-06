extends Node2D

@export_category("Simulation parameters")
@export var N: int = 1000
@export var coupling: float = 6
@export var frequency_sigma:  float = 2
@export var frequency_mean: float = 0.

@export var dt: float = 1./60 # 60 fps
@export var sub_steps: int = 1
@export var solver: KuramotoSolver
@export var integrator_idx: int = 0

@export_category("Visual parameters")
@export var entity_color: Color = Color.RED
@export var entity_size: float = 2.

# positions
var positions := []
var frame_time: float = 1./60

func _ready() -> void:	
	generate_positions()
	solver.initialize()

func _process(delta: float) -> void:
	%FPSLabel.text = "FPS: " + str(Engine.get_frames_per_second())

	frame_time = Time.get_ticks_usec()
	solver.simulation_step()
	frame_time = Time.get_ticks_usec() - frame_time
	
	%FrameTimeLabel.text = "Delta (ms): " + str(snappedf(frame_time/1000, 1e-3))	
	queue_redraw()

func _draw() -> void:
	solver.on_draw()

func generate_positions():
	positions.resize(N)
	for i in N: 
		positions[i] = Vector2(randf_range(0, get_viewport().size.x), randf_range(0, get_viewport().size.y))

func _on_point_slider_drag_ended(value_changed: bool) -> void:
	%PointLabel.text = "Fireflies: " + str(%PointSlider.value)
	N = %PointSlider.value
	generate_positions()
	solver.initialize()

func _on_coupling_slider_drag_ended(value_changed: bool) -> void:
	%CouplingLabel.text = "Coupling: " + str(%CouplingSlider.value)
	coupling = %CouplingSlider.value

func _on_mean_slider_drag_ended(value_changed: bool) -> void:
	%MeanLabel.text = "Frequency mean: " + str(%MeanSlider.value)
	solver.generate_frequencies()

func _on_std_slider_drag_ended(value_changed: bool) -> void:
	%StdLabel.text = "Frequency std: " + str(%StdSlider.value)
	solver.generate_frequencies()

func _on_restart_button_pressed() -> void:
	solver.initialize()

func _on_substep_slider_drag_ended(value_changed: bool) -> void:
	%SubstepLabel.text = "Sub-steps: " + str(%SubstepSlider.value)
	sub_steps = %SubstepSlider.value

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	solver.initialize()

func _on_integrator_option_item_selected(index: int) -> void:
	integrator_idx = index
