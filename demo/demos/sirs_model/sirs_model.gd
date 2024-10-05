extends Node2D

@export_category("Simulation parameters")
@export var solver: SIRSolver

@export var N: int = 20
@export var tauI: int = 4
@export var tauR: int = 6
@onready var tau0 := tauI + tauR

@export_category("Visual parameters")
@export var num_draw_points: int = 50
@export var draw_space = 5
@export var colors = []

var frame_time: float = 0.

func _ready() -> void:
	%PointSlider.value = N
	%PointLabel.text = "Grid: " + str(N) + "x" + str(N)
	
	%InfectedSlider.value = tauI
	%InfectedLabel.text = str("Infection time: " + str(tauI))
	
	%RecoverySlider.value = tauR
	%RecoveryLabel.text = str("Recovery time: " + str(tauR))
	
	generate_colors()	
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

func generate_colors() -> void:
	colors = []
	colors.push_back(Color.WHITE)
	for i in tauI: colors.push_back(Color.RED)
	for i in tauR: colors.push_back(Color.ORANGE)
	colors.push_back(Color.BLACK)

func _on_point_slider_drag_ended(value_changed: bool) -> void:
	N = %PointSlider.value
	%PointLabel.text = "Grid: " + str(N) + "x" + str(N)
	solver.initialize()

func _on_infected_slider_drag_ended(value_changed: bool) -> void:
	tauI = %InfectedSlider.value
	%InfectedLabel.text = str("Infection time: " + str(tauI))
	tau0 = tauI + tauR
	solver.initialize()
	generate_colors()

func _on_recovery_slider_drag_ended(value_changed: bool) -> void:
	tauR = %RecoverySlider.value
	%RecoveryLabel.text = str("Recovery time: " + str(tauR))
	tau0 = tauI + tauR
	solver.initialize()
	generate_colors()

func _on_restart_button_pressed() -> void:
	solver.initialize()

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	solver.initialize()
