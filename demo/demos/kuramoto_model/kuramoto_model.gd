extends Node2D

@export_category("Simulation parameters")
@export var N: int = 1000
@export var coupling: float = 1
@export var frequency_sigma:  float = 0.2
@export var frequency_mean: float = 2.

@export var dt: float = 1./60 # 60 fps
@export var sub_steps: int = 1
@export var solver: KuramotoSolver
@export var integrator_idx: int = 0

# positions
var positions := []
var frame_time: float = 1./60

# particle shader
@onready var fireflies := $Fireflies
var TEXTURE_SIZE: int
var fireflies_data: Image
var fireflies_texture: ImageTexture

func _ready() -> void:
	%PointSlider.value = N
	%PointLabel.text = "Fireflies: " + str(%PointSlider.value)
	%CouplingSlider.value = coupling
	%CouplingLabel.text = "Coupling: " + str(%CouplingSlider.value)
	%MeanSlider.value = frequency_mean
	%MeanLabel.text = "Frequency mean: " + str(%MeanSlider.value)
	%StdSlider.value = frequency_sigma
	%StdLabel.text = "Frequency std: " + str(%StdSlider.value)

	_restart_simulation()

func _process(delta: float) -> void:
	%FPSLabel.text = "FPS: " + str(Engine.get_frames_per_second())

	frame_time = Time.get_ticks_usec()
	solver.simulation_step()
	frame_time = Time.get_ticks_usec() - frame_time

	%FrameTimeLabel.text = "Delta (ms): " + str(snappedf(frame_time/1000, 1e-3))
	solver.on_draw()

#func _draw() -> void:
	#solver.on_draw()

func _on_point_slider_drag_ended(value_changed: bool) -> void:
	%PointLabel.text = "Fireflies: " + str(%PointSlider.value)
	N = %PointSlider.value
	_restart_simulation()
	_randomize_positions()

func _on_coupling_slider_drag_ended(value_changed: bool) -> void:
	%CouplingLabel.text = "Coupling: " + str(%CouplingSlider.value)
	coupling = %CouplingSlider.value

func _on_mean_slider_drag_ended(value_changed: bool) -> void:
	%MeanLabel.text = "Frequency mean: " + str(%MeanSlider.value)
	frequency_mean = %MeanSlider.value
	solver.generate_frequencies()

func _on_std_slider_drag_ended(value_changed: bool) -> void:
	%StdLabel.text = "Frequency std: " + str(%StdSlider.value)
	frequency_sigma = %StdSlider.value
	solver.generate_frequencies()

func _on_restart_button_pressed() -> void:
	_restart_simulation()
	_randomize_positions()

func _on_substep_slider_drag_ended(value_changed: bool) -> void:
	%SubstepLabel.text = "Sub-steps: " + str(%SubstepSlider.value)
	sub_steps = %SubstepSlider.value

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	_restart_simulation()

func _on_integrator_option_item_selected(index: int) -> void:
	integrator_idx = index

func _restart_simulation() -> void:
	_generate_fireflies()
	solver.initialize()

func _generate_fireflies() -> void:
	TEXTURE_SIZE = int(ceil(sqrt(N)))
	fireflies_data = Image.create(TEXTURE_SIZE, TEXTURE_SIZE, false, Image.FORMAT_RGBAH)
	fireflies_texture = ImageTexture.create_from_image(fireflies_data)

	fireflies.amount = N
	fireflies.process_material.set_shader_parameter("fireflies_data", fireflies_texture)

	positions.resize(N)
	for i in N:
		positions[i] = Vector2(randf_range(0, get_viewport_rect().size.x), randf_range(0, get_viewport_rect().size.y))

func _randomize_positions() -> void:
	for i in N:
		positions[i].x = randf_range(0, get_viewport_rect().size.x)
		positions[i].y = randf_range(0, get_viewport_rect().size.y)

func set_alphas(alphas: Variant) -> void:
	if alphas is Array:
		for i in alphas.size():
			var pixel_pos := Vector2(int(i / TEXTURE_SIZE), int(i % TEXTURE_SIZE))
			fireflies_data.set_pixel(
				pixel_pos.x, pixel_pos.y,
				Color(positions[i].x, positions[i].y, sin(alphas[i])/2 + 0.5, 0)
				)

	elif alphas is NDArray:
		for i in alphas.size():
			var pixel_pos := Vector2(int(i / TEXTURE_SIZE), int(i % TEXTURE_SIZE))
			fireflies_data.set_pixel(
				pixel_pos.x, pixel_pos.y,
				Color(positions[i].x, positions[i].y, alphas.get_float(i), 0)
				)

	fireflies_texture.update(fireflies_data)
