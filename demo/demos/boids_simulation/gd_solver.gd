extends BoidsSolver

# parameters (speed, boids, etc.) defined in params
# params is set to BoidsModel and can be modified in Editor and in boids_model.gd


@export var update_interval: int = 5000  # Number of frames between angle updates
var frame_counter: int = 0
var new_direction = Vector2.ONE

func initialize() -> void:#
	
	
	
	
	# create position vector and initialize with random Vector2s inside screen
		
	# create velocity vector and initialize with Vector2s of same value in random directions

	pass

func simulation_step(delta: float, velocity: Vector2, speed: float,
 visual_range: float, separation_weight: float, alignment_weight: float,
 cohesion_weight: float, position: Vector2, boid_sprite: Sprite2D, boids: Array) -> void:

	
	# for each boid collect others in visual range
	# calculate new velocity directions according to:
		# Seperation
		# Alignment
		# Cohesion#

	var separation_range = 50.0
	

	for boid in boids:
		var separation = apply_separation(boid, boids, separation_range)
		var alignment = apply_alignment(boid, boids, visual_range)
		var cohesion = apply_cohesion(boid, boids, visual_range)

		# Combine the three steering behaviors
		var steer = separation * separation_weight + alignment * alignment_weight + cohesion * cohesion_weight

		# Blend with current velocity and normalize to maintain speed
		boid.velocity = (boid.velocity + steer).normalized() * speed
		# (Additional noise)
	# Apply a random angle noise every n frames
	
	update_boids_noise(boids, speed, delta)
		
	
	frame_counter += 1
	
	
	

	
	

	pass
	
func apply_separation(boid, boids: Array, separation_distance: float) -> Vector2:
	var steer = Vector2.ZERO
	var total = 0

	for other in boids:
		if other == boid:
			continue

		var dist = boid.node.position.distance_to(other.node.position)

		if dist < separation_distance and dist > 0:
			# Vector pointing away from the neighbor, inversely weighted by distance
			var diff = (boid.node.position - other.node.position).normalized() / dist
			steer += diff
			total += 1

	if total > 0:
		steer /= total
		steer = steer.normalized()
	
	return steer
	
func apply_alignment(boid: Dictionary, boids: Array, visual_range: float) -> Vector2:
	var avg_velocity = Vector2.ZERO
	var count = 0

	for other in boids:
		if other == boid:
			continue

		var dist = boid.node.position.distance_to(other.node.position)
		if dist < visual_range:
			avg_velocity += other.velocity
			count += 1

	if count > 0:
		avg_velocity /= count
		avg_velocity = avg_velocity.normalized()
		return avg_velocity
	else:
		return Vector2.ZERO
		
func apply_cohesion(boid: Dictionary, boids: Array, visual_range: float) -> Vector2:
	var center = Vector2.ZERO
	var count = 0

	for other in boids:
		if other == boid:
			continue

		var dist = boid.node.position.distance_to(other.node.position)
		if dist < visual_range:
			center += other.node.position
			count += 1

	if count > 0:
		center /= count
		var direction = (center - boid.node.position).normalized()
		return direction
	else:
		return Vector2.ZERO


func update_boids_noise(boids: Array, speed: float, delta: float) -> void:
	if frame_counter >= update_interval:
		frame_counter = 0
		for boid in boids:
			
				
				
			# Apply a small random variation to the current angle
			var angle_variation = randf_range(-0.5, 0.5)
			var adjusted_angle = boid.velocity.angle() + angle_variation
			new_direction = Vector2(cos(adjusted_angle), sin(adjusted_angle))
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
	


	
