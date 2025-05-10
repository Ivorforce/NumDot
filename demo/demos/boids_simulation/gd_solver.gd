extends BoidsSolver

# parameters (speed, boids, etc.) defined in params
# params is set to BoidsModel and can be modified in Editor and in boids_model.gd

var position: Vector2 = Vector2(500, 500) # position vector
#var velocity: Vector2 = Vector2.ZERO

var texture = preload("res://demos/boids_simulation/boid.png")

@onready var boid_sprite: Sprite2D = $/root/BoidsModel/Sprite2D

func initialize() -> void:#
	
		# Load the boid sprite
	
	boid_sprite.texture = texture
	
	# Set the initial position and size
	boid_sprite.scale = Vector2(0.1, 0.1)
	boid_sprite.position = position
	
	
	# create position vector and initialize with random Vector2s inside screen
		
	# create velocity vector and initialize with Vector2s of same value in random directions

	pass

func simulation_step(delta: float) -> void:
	# for each boid collect others in visual range
	# calculate new velocity directions according to:
		# Seperation
		# Alignment
		# Cohesion
		# (Additional noise)
	# apply velocities to positions

	pass

func update_boids() -> void:
	# apply position from position vector to each boid
	# derive and apply rotation from velocity vector direction to each boid

	pass
	
