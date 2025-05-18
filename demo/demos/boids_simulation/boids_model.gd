extends Node2D

@export var boid_count: int = 20
@export var speed: float = 200.0  # Pixels per second
@export var range: float = 100.0
@export var separation_weight: float = 1.0
@export var alignment_weight: float = 0.5
@export var cohesion_weight: float = 0.5
@export var noise_weight: float = 0.5

@export var scale_factor: float = 0.1

@export_category("Simulation parameters")
@export var solver: BoidsSolver

#var position = Vector2(500, 500) # position vector
var velocity: Vector2 = Vector2(100,50)


var texture = preload("res://demos/boids_simulation/boid.tres")

@onready var boid_sprite: Sprite2D = $/root/BoidsModel/Sprite2D

#TODO Implement reset

func _ready() -> void:
	# Create Boids container if it does not exist
	if not has_node("Boids"):
		var boids_container = Node2D.new()
		boids_container.name = "Boids"
		add_child(boids_container)

	# Create boids as children of the Boids container
	initialize_boids(boid_count)

	solver.initialize()
		##TODO initialize in model?
		## Initialize boid data
		#var boid_data = {
			#"node": boid,
			##Assign same speed and random direction based on rotation
			#"velocity": Vector2(cos(boid.rotation), sin(boid.rotation)).normalized() * speed
		#}
		#boids.append(boid_data)
	# Set correct values in GUI

	# Initialize boids as scenes from res://demos/boids_simulation/boid.tscn
		# Load the boid sprite

	# TODO Add border around screen redirecting boids to screen center


func _process(delta: float) -> void:
	%FPSLabel.text = "FPS: " + str(Engine.get_frames_per_second())

	solver.simulation_step(delta)

	# Call simulation_step(delta)
	# TODO delta display
	# Update GUI


func initialize_boids(target_count: int) -> void:
	var current_count = $Boids.get_child_count()
	if current_count > target_count:
		var boids = $Boids.get_children()
		for boid in range(target_count, current_count).map(func(index): return boids[index]):
			boid.queue_free()
	else:
		for i in range(target_count-current_count):
			add_boid(current_count+i)


func add_boid(i: int):
	var boid = Sprite2D.new()
	boid.texture = texture
	boid.set_modulate(Color.DARK_SLATE_GRAY)
	boid.z_index = -1
	boid.scale = Vector2(0.1, 0.1)
	boid.name = "Boid" + str(i)  # Naming each boid
	$Boids.add_child(boid)


func wrap_around(boids: Array):
	return
	var screen_size = get_viewport_rect().size
	for boid in boids:

		# TODO in solver, add offset?
		# Check horizontal boundaries
		if boid.node.position.x > screen_size.x:
			boid.node.position.x = 0
		elif boid.node.position.x < 0:
			boid.node.position.x = screen_size.x

		# Check vertical boundaries
		if boid.node.position.y > screen_size.y:
			boid.node.position.y = 0
		elif boid.node.position.y < 0:
			boid.node.position.y = screen_size.y


func _on_speed_slider_value_changed(value) -> void:
	speed = value
	%SpeedLabel.text = "Speed: " + str(speed)

func _on_range_slider_value_changed(value) -> void:
	range = value
	%RangeLabel.text = "Range: " + str(range)

func _on_separation_slider_value_changed(value) -> void:
	separation_weight = value
	%SeparationLabel.text = "Separation: " + str(separation_weight)

func _on_alignment_slider_value_changed(value) -> void:
	alignment_weight = value
	%AlignmentLabel.text = "Alignment: " + str(alignment_weight)

func _on_cohesion_slider_value_changed(value) -> void:
	cohesion_weight = value
	%CohesionLabel.text = "Cohesion: " + str(cohesion_weight)

func _on_noise_slider_value_changed(value) -> void:
	noise_weight = value
	%NoiseLabel.text = "Noise: " + str(noise_weight)

func _on_restart_button_pressed() -> void:
	solver.initialize()

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	solver.initialize()

func _on_number_of_boids_slider_value_changed(value) -> void:
	boid_count = value
	%NumberOfBoids.text = "Boids: " + str(boid_count)
	initialize_boids(boid_count)
