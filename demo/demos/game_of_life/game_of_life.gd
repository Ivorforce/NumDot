extends Node2D

@export_category("Simulation parameters")
@export var solver: GameOfLifeSolver

@export var N: int = 20

@export var steps_per_second: float = 1.0
var current_step: float = 0.0

@export_category("Visual parameters")
@export var color_on: Color = Color.WHITE
@export var color_off: Color = Color.BLACK

@export var texture_rect: TextureRect
var _image: Image

var frame_time: float = 0.

func _ready() -> void:
	%PointSlider.value = N
	%PointLabel.text = "Grid: " + str(N) + "x" + str(N)
	
	resize_image()
	
	solver.initialize()

func _process(delta: float) -> void:
	%FPSLabel.text = "FPS: " + str(Engine.get_frames_per_second())

	var next_step := current_step + delta * steps_per_second
	for i in (int(next_step) - int(current_step)):
		frame_time = Time.get_ticks_usec()
		solver.simulation_step()
		frame_time = Time.get_ticks_usec() - frame_time
	current_step = next_step
	
	%FrameTimeLabel.text = "Delta (ms): " + str(snappedf(frame_time/1000, 1e-3))
	solver.on_draw()

func _on_point_slider_drag_ended(value_changed: bool) -> void:
	N = %PointSlider.value
	%PointLabel.text = "Grid: " + str(N) + "x" + str(N)
	resize_image()
	solver.initialize()

func _on_steps_per_second_slider_drag_ended(value_changed: bool) -> void:
	steps_per_second = %StepsPerSecondSlider.value
	%StepsPerSecondLabel.text = "Speed: %.2f" % steps_per_second

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
	texture_rect.set_size(Vector2(500, 500))
	var origin = get_viewport_rect().get_center() - texture_rect.size/2
	texture_rect.position = origin
