extends Node

@export var params: Node2D


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
	# Add border around screen redirecting boids to screen center

	#solver.initialize()
	pass

func _process(delta: float) -> void:
	# Call simulation_step(delta)
	# Update GUI

	pass


# TODO handle GUI inputs
