extends Node2D

@export var boid_count: int = 20
@export var speed: float = 200.0  # Pixels per second
@export var visual_range: float = 100.0
@export var separation_weight: float = 1.0
@export var alignment_weight: float = 0.5
@export var cohesion_weight: float = 0.5

# Store data for each sprite
var boids = []
var boids_container: Node2D

@export_category("Simulation parameters")
@export var solver: BoidsSolver

#var position = Vector2(500, 500) # position vector
var velocity: Vector2 = Vector2(100,50)


var texture = preload("res://demos/boids_simulation/boid.png")

@onready var boid_sprite: Sprite2D = $/root/BoidsModel/Sprite2D

#TODO Implement reset

func _ready() -> void:
	# Create Boids container if it does not exist
	if not has_node("Boids"):
		boids_container = Node2D.new()
		boids_container.name = "Boids"
		add_child(boids_container)
	else:
		boids_container = $Boids
		
		# Create boids as children of the Boids container
	for i in range(boid_count):
		var boid = Sprite2D.new()
		boid.texture = texture
		#Assign a boid a random position and standard size
		boid.scale = Vector2(0.1, 0.1)
		boid.position = Vector2(randf_range(0, get_viewport_rect().size.x), randf_range(0, get_viewport_rect().size.y))
		# Assign a random start rotation (in radians)
		boid.rotation = randf_range(0, PI * 2)
		boid.name = "Boid " + str(i + 1)  # Naming each boid
		
		boids_container.add_child(boid)
		
		

		# Initialize boid data
		var boid_data = {
			"node": boid,
			#Assign same speed and random direction based on rotation
			"velocity": Vector2(cos(boid.rotation), sin(boid.rotation)).normalized() * speed
		}
		boids.append(boid_data)
	# Set correct values in GUI

	# Initialize boids as scenes from res://demos/boids_simulation/boid.tscn
		# Load the boid sprite
	
	
	# Add border around screen redirecting boids to screen center
	

	#TODO fix workaround below 
	self.solver = $Solvers/GDSolver
	solver.initialize()
	pass

func _process(delta: float) -> void:
	%FPSLabel.text = "FPS: " + str(Engine.get_frames_per_second())
	# Call simulation_step(delta)
	#TODO delta display
	solver.simulation_step(delta, velocity, speed, visual_range, separation_weight, alignment_weight, cohesion_weight, position, boid_sprite, boids)
	wrap_around(boids)
	# Update GUI

	pass
	
func initialize_boids(target_count: int) -> void:
	# Clear existing boids
	for boid_data in boids:
		boid_data["node"].queue_free()
	boids.clear()

	# Create new boids
	for i in range(target_count):
		add_boid(i)
		
func add_boid(i: int):
	var boid = Sprite2D.new()
	boid.texture = texture
	#Assign a boid a random position and standard size
	boid.scale = Vector2(0.1, 0.1)
	boid.position = Vector2(randf_range(0, get_viewport_rect().size.x), randf_range(0, get_viewport_rect().size.y))
	# Assign a random start rotation (in radians)
	boid.rotation = randf_range(0, PI * 2)
	boid.name = "Boid " + str(i + 1)  # Naming each boid
	
	boids_container.add_child(boid)
	
	

	# Initialize boid data
	var boid_data = {
		"node": boid,
		#Assign same speed and random direction based on rotation
		"velocity": Vector2(cos(boid.rotation), sin(boid.rotation)).normalized() * speed
	}
	boids.append(boid_data)
	
func wrap_around(boids: Array):
	var screen_size = get_viewport_rect().size
	for boid in boids:
		

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

func _on_speed_slider_drag_ended(value_changed: bool) -> void:
	if value_changed:
		speed = %SpeedSlider.value
		%SpeedLabel.text = "Speed: " + str(round(speed))
		
func _on_visual_range_slider_drag_ended(value_changed: bool) -> void:
	if value_changed:
		visual_range = %VisualRangeSlider.value
		%VisualRangeLabel.text = "Visual Range: " + str(round(visual_range))
		
func _on_separation_slider_drag_ended(value_changed: bool) -> void:
	if value_changed:
		separation_weight = %SeparationSlider.value
		%SeparationLabel.text = "Separation: " + str(snapped(separation_weight, 0.1))
		
func _on_alignment_slider_drag_ended(value_changed: bool) -> void:
	if value_changed:
		alignment_weight = %AlignmentSlider.value
		%AlignmentLabel.text = "Alignment: " + str(snapped(alignment_weight, 0.1))
		
func _on_cohesion_slider_drag_ended(value_changed: bool) -> void:
	if value_changed:
		cohesion_weight = %CohesionSlider.value
		%CohesionLabel.text = "Cohesion: " + str(snapped(cohesion_weight, 0.1))
		
func _on_restart_button_pressed() -> void:
	solver.initialize()

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	solver.initialize()
	
func _on_number_of_boids_slider_drag_ended(value_changed: bool) -> void:
	boid_count = %NumberOfBoidsSlider.value
	%NumberOfBoids.text = "Boids: "+str(boid_count)
	initialize_boids(boid_count)
	
