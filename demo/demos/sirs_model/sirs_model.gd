extends Node2D

@export_category("Simulation parameters")
@export var solver: SIRSolver

@export var N: int = 20
@export var spread: float = 1.0
@export var tauI: int = 4
@export var tauR: int = 6
@onready var tau0 := tauI + tauR

@export_category("Visual parameters")
@export var num_draw_points: int = 50
@export var draw_scale = 5
@export var colors = []

@export var color_susceptible: Color = Color.WHITE
@export var color_infected: Color = Color.RED
@export var color_recovered: Color = Color.ORANGE

@export var texture_rect: TextureRect
var _image: Image

var frame_time: float = 0.

func _ready() -> void:
	%PointSlider.value = N
	%PointLabel.text = "Grid: " + str(N) + "x" + str(N)

	%SpreadSlider.value = spread
	%SpreadLabel.text = "Spread: %.3f" % spread
	
	%InfectedSlider.value = tauI
	%InfectedLabel.text = "Infection time: " + str(tauI)
	
	%RecoverySlider.value = tauR
	%RecoveryLabel.text = "Recovery time: " + str(tauR)
	
	generate_colors()	
	resize_image()
	create_legend()
	texture_rect.scale = Vector2(draw_scale, draw_scale)
	
	solver.initialize()

func _process(delta: float) -> void:

	%FPSLabel.text = "FPS: " + str(Engine.get_frames_per_second())

	frame_time = Time.get_ticks_usec()
	solver.simulation_step()
	frame_time = Time.get_ticks_usec() - frame_time
	
	%FrameTimeLabel.text = "Delta (ms): " + str(snappedf(frame_time/1000, 1e-3))
	solver.on_draw()

func generate_colors() -> void:
	colors = []
	colors.push_back(color_susceptible)
	for i in tauI: colors.push_back(color_infected)
	for i in tauR: colors.push_back(color_recovered)
	colors.push_back(Color.BLACK)

func _on_point_slider_drag_ended(value_changed: bool) -> void:
	N = %PointSlider.value
	%PointLabel.text = "Grid: " + str(N) + "x" + str(N)
	resize_image()
	solver.initialize()

func _on_spread_slider_drag_ended(value_changed: bool) -> void:
	spread = %SpreadSlider.value
	%SpreadLabel.text = "Spread: %.3f" % spread
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

func update_texture() -> void:
	texture_rect.texture.update(_image)
	
func resize_image() -> void:
	_image = Image.create(N, N, false, Image.FORMAT_RGBA8)

	texture_rect.texture = ImageTexture.create_from_image(_image)
	texture_rect.set_size(Vector2(N, N))
	var origin = get_viewport_rect().get_center() - texture_rect.size/2 * draw_scale
	texture_rect.position = origin

func create_legend() -> void:
	var _img = Image.create(1, 1, false, Image.FORMAT_RGBA8)
	
	# infected
	_img.set_pixel(0, 0, color_infected)
	$LegendInfected.texture = ImageTexture.create_from_image(_img)
	
	# recovered
	_img.set_pixel(0, 0, color_recovered)
	$LegendRecovered.texture = ImageTexture.create_from_image(_img)
