extends Node2D

@export var boid_count: int = 5
# Store data for each sprite
var boids = []
var boids_container: Node2D

@export_category("Simulation parameters")
@export var solver: BoidsSolver

#var position = Vector2(500, 500) # position vector
var velocity: Vector2 = Vector2(100,50)


var texture = preload("res://demos/boids_simulation/boid.png")

@onready var boid_sprite: Sprite2D = $/root/BoidsModel/Sprite2D

@export var speed: float = 200.0  # Pixels per second




@export var N: int = 20
# Implement slider functionalities for:
	# Number of Boids
	# Speed
	# Visual range
	# Seperation
	# Alignment
	# Cohesion

# Implement reset

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
	
	boid_sprite.texture = texture
	
	# Set the initial position and size
	boid_sprite.scale = Vector2(0.1, 0.1)
	boid_sprite.position = position
	# Add border around screen redirecting boids to screen center
	

	#TODO fix workaround below 
	self.solver = $Solvers/GDSolver
	solver.initialize()
	pass

func _process(delta: float) -> void:
	# Call simulation_step(delta)
	
	solver.simulation_step(delta, velocity, speed, position, boid_sprite, boids)
	wrap_around(boids)
	# Update GUI

	pass
	
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

func _on_point_slider_drag_ended(value_changed: bool) -> void:
	N = %PointSlider.value
	%PointLabel.text = "Grid: " + str(N) + "x" + str(N)
	#resize_image()
	solver.initialize()

#func _on_steps_per_second_slider_drag_ended(value_changed: bool) -> void:
	#steps_per_second = %StepsPerSecondSlider.value
	#%StepsPerSecondLabel.text = "Speed: %.2f" % steps_per_second

func _on_restart_button_pressed() -> void:
	solver.initialize()

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	solver.initialize()
# TODO handle GUI inputs
