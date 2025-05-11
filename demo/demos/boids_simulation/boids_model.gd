extends Node2D


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
	solver.simulation_step(delta, velocity, speed, position, boid_sprite)
	wrap_around()
	# Update GUI

	pass
	
func wrap_around():
	var screen_size = get_viewport_rect().size

	# Check horizontal boundaries
	if boid_sprite.position.x > screen_size.x:
		boid_sprite.position.x = 0
	elif boid_sprite.position.x < 0:
		boid_sprite.position.x = screen_size.x

	# Check vertical boundaries
	if boid_sprite.position.y > screen_size.y:
		boid_sprite.position.y = 0
	elif boid_sprite.position.y < 0:
		boid_sprite.position.y = screen_size.y

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
