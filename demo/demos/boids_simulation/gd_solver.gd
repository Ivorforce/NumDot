extends BoidsSolver

# parameters (speed, boids, etc.) defined in params
# params is set to BoidsModel and can be modified in Editor and in boids_model.gd


@export var update_interval: int = 50  # Number of frames between angle updates
var frame_counter: int = 0
var new_direction = Vector2.ONE

func initialize() -> void:#
	
	
	
	
	# create position vector and initialize with random Vector2s inside screen
		
	# create velocity vector and initialize with Vector2s of same value in random directions

	pass

func simulation_step(delta: float, velocity: Vector2, speed: float, position: Vector2, boid_sprite: Sprite2D, boids: Array) -> void:

	
	# for each boid collect others in visual range
	# calculate new velocity directions according to:
		# Seperation
		# Alignment
		# Cohesion
		# (Additional noise)
	# Apply a random angle noise every n frames
	
	update_boids(boids, speed, delta)
		
	
	frame_counter += 1
	
	
	

	
	

	pass

func update_boids(boids: Array, speed: float, delta: float) -> void:
	if frame_counter >= update_interval:
		frame_counter = 0
		for boid in boids:
			
				
				
			# Apply a small random variation to the current angle
			var angle_variation = randf_range(-0.01, 0.01)
			var new_angle = boid.node.rotation + angle_variation
			new_direction = Vector2(cos(new_angle), sin(new_angle))
			boid.velocity = new_direction * speed
			# apply velocities to positions
			
			boid.node.position += boid.velocity * delta
			
			#offset angle for right facing direction
			boid.node.rotation = boid.velocity.angle()  + PI / 2 
			# apply position from position vector to each boid
			# derive and apply rotation from velocity vector direction to each boid
	else:
		for boid in boids:
			# apply velocities to positions
			
			boid.node.position += boid.velocity * delta
			
			#offset angle for right facing direction
			boid.node.rotation = boid.velocity.angle()  + PI / 2 
			# apply position from position vector to each boid
			# derive and apply rotation from velocity vector direction to each boid

	pass
	


	
