extends BoidsSolver

# parameters (speed, boids, etc.) defined in params
# params is set to BoidsModel and can be modified in Editor and in boids_model.gd

@export var rotation_speed: float = 2.0  # Higher value = faster rotation
@export var update_interval: int = 5  # Number of frames between angle updates
var frame_counter: int = 0
var noise_angle = randf_range(-0.5, 0.5) # deafult random noise angle

func initialize() -> void:#
	
	
	
	
	# create position vector and initialize with random Vector2s inside screen
		
	# create velocity vector and initialize with Vector2s of same value in random directions

	pass

func simulation_step(delta: float, velocity: Vector2, speed: float, position: Vector2, boid_sprite: Sprite2D) -> void:
	
	
	
	# for each boid collect others in visual range
	# calculate new velocity directions according to:
		# Seperation
		# Alignment
		# Cohesion
		# (Additional noise)
	# Apply a random angle noise every n frames
	frame_counter += 1
	if frame_counter >= update_interval:
		frame_counter = 0
		noise_angle += randf_range(-0.1, 0.1)
	
	var new_direction = velocity.rotated(noise_angle).normalized()
	velocity = new_direction * speed
	# apply velocities to positions
	
	boid_sprite.position += velocity * delta
	
	#offset angle for right facing direction
	boid_sprite.rotation = velocity.angle()  + PI / 2 

	
	

	pass

func update_boids() -> void:
	# apply position from position vector to each boid
	# derive and apply rotation from velocity vector direction to each boid

	pass
	


	
