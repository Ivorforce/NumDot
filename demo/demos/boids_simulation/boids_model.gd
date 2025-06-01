extends Node2D

@export var boid_count: int = 20
@export var speed: float = 200.0
@export var range: float = 100.0
@export var separation_weight: float = 1.0
@export var alignment_weight: float = 0.5
@export var cohesion_weight: float = 0.5

@export var scale_factor: float = 0.1

@export_category("Simulation parameters")
@export var solver: BoidsSolver

var texture = preload("res://demos/boids_simulation/boid.tres")

var frame_time: float = 0.

func _ready() -> void:
	"""
	Synchronize initial values with GUI, instantiate boid nodes, and initialize solver.
	"""
	# Synchronize initial values with GUI
	boid_count = %NumberOfBoidsSlider.value
	speed = %SpeedSlider.value
	range = %RangeSlider.value
	separation_weight = %SeparationSlider.value
	alignment_weight = %AlignmentSlider.value
	cohesion_weight = %CohesionSlider.value

	# Instantiate Boid nodes under $Boids
	if not has_node("Boids"):
		var boids_container = Node2D.new()
		boids_container.name = "Boids"
		add_child(boids_container)
	initialize_boids(boid_count)

	solver.initialize()


func _process(delta: float) -> void:
	"""
	Called every frame. Performs simulation step and updates GUI metrics.
	
	Parameters:
	delta (float): Time since last frame in seconds.
	"""
	frame_time = Time.get_ticks_usec()
	solver.simulation_step(delta)
	frame_time = Time.get_ticks_usec() - frame_time

	%FPSLabel.text = "FPS: " + str(Engine.get_frames_per_second())
	%FrameTimeLabel.text = "Delta (ms): " + str(snappedf(frame_time/1000, 1e-3))


func initialize_boids(target_count: int) -> void:
	"""
	Initializes boid nodes to match the specified target count.
	
	Parameters:
	target_count (int): Desired number of boids.
	"""
	var current_count = $Boids.get_child_count()
	if current_count > target_count:
		var boids = $Boids.get_children()
		for boid in range(target_count, current_count).map(func(index): return boids[index]):
			boid.queue_free()
	else:
		for i in range(target_count-current_count):
			add_boid(current_count+i)


func add_boid(i: int):
	"""
	Adds a single boid sprite to the simulation.
	
	Parameters:
	i (int): Index identifier for naming the new boid.
	"""
	var boid = Sprite2D.new()
	boid.texture = texture
	boid.set_modulate(Color.DARK_SLATE_GRAY)
	boid.z_index = -1
	boid.scale *= scale_factor
	boid.name = "Boid" + str(i)
	$Boids.add_child(boid)


func _on_speed_slider_value_changed(value) -> void:
	"""
	Updates speed based on slider input and updates GUI label.
	
	Parameters:
	value (float): New speed value from the slider.
	"""
	speed = value
	%SpeedLabel.text = "Speed: " + str(speed)

func _on_range_slider_value_changed(value) -> void:
	"""
	Updates range based on slider input and updates GUI label.
	
	Parameters:
	value (float): New range value from the slider.
	"""
	range = value
	%RangeLabel.text = "Range: " + str(range)

func _on_separation_slider_value_changed(value) -> void:
	"""
	Updates separation weight based on slider input and updates GUI label.
	
	Parameters:
	value (float): New separation weight value.
	"""
	separation_weight = value
	%SeparationLabel.text = "Separation: " + str(separation_weight)

func _on_alignment_slider_value_changed(value) -> void:
	"""
	Updates alignment weight based on slider input and updates GUI label.
	
	Parameters:
	value (float): New alignment weight value.
	"""
	alignment_weight = value
	%AlignmentLabel.text = "Alignment: " + str(alignment_weight)

func _on_cohesion_slider_value_changed(value) -> void:
	"""
	Updates cohesion weight based on slider input and updates GUI label.
	
	Parameters:
	value (float): New cohesion weight value.
	"""
	cohesion_weight = value
	%CohesionLabel.text = "Cohesion: " + str(cohesion_weight)

func _on_restart_button_pressed() -> void:
	"""
	Resets and reinitializes the solver upon pressing the restart button.
	"""
	solver.initialize()

func _on_solver_option_item_selected(index: int) -> void:
	"""
	Switches solver based on selection from options.
	
	Parameters:
	index (int): Index of the newly selected solver.
	"""
	solver = $Solvers.get_child(index)
	solver.initialize()

func _on_number_of_boids_slider_value_changed(value) -> void:
	"""
	Adjusts the number of boids based on slider input and updates GUI label.
	
	Parameters:
	value (int): New desired count of boids.
	"""
	boid_count = value
	%NumberOfBoids.text = "Boids: " + str(boid_count)
	initialize_boids(boid_count)
