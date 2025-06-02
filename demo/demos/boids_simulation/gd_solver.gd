extends BoidsSolver

var positions: Array[Vector2]
var directions: Array[Vector2]

var screen_size: Vector2

func initialize() -> void:
	"""
	Initializes the solver by setting the screen size
	and generating initial position and direction arrays.
	"""
	screen_size = params.get_viewport_rect().size

	# Initialize position and direction vector
	positions = initialize_position_array(params.boid_count)
	directions = initialize_direction_array(params.boid_count)


# Helper function to create position direction vector with length
func initialize_position_array(length: int) -> Array[Vector2]:
	"""
	Creates array of random positions within screen bounds.

	Parameters:
	length (int): Number of positions.

	Returns:
	Array[Vector2]: Random position vectors.
	"""
	# Initialize position Array with |length| Vector2s
	# Values are random 2D-positions on screen
	var positions_xy: Array[Vector2] = []
	for i in range(length):
		positions_xy.append(Vector2(randf()*screen_size.x, randf()*screen_size.y))
	return positions_xy


# Helper function to create random direction vector with length
func initialize_direction_array(length: int) -> Array[Vector2]:
	"""
	Creates array of normalized random direction vectors.

	Parameters:
	length (int): Number of directions.

	Returns:
	Array[Vector2]: Random direction vectors.
	"""
	# Initialize direction Array with |length| Vector2s
	# Values are normalized 2D-vectors with random angle
	var directions_xy: Array[Vector2] = []
	for i in range(length):
		var angle = 2*PI*randf()
		var direction = Vector2(cos(angle), sin(angle))
		directions_xy.append(direction)
	return directions_xy


func simulation_step(delta: float) -> void:
	"""
	Updates positions and directions based on separation,
	alignment, and cohesion during simulation.

	Parameters:
	delta (float): Time since last frame.
	"""
	# Check if boid_count has been changed, update vector sizes accordingly
	var boid_count_difference = params.boid_count-positions.size()
	if boid_count_difference < 0:
		positions.resize(params.boid_count)
		directions.resize(params.boid_count)
	elif boid_count_difference > 0:
		var new_positions := initialize_position_array(boid_count_difference)
		var new_directions := initialize_direction_array(boid_count_difference)
		positions.append_array(new_positions)
		directions.append_array(new_directions)

	# Calculate separation, alignment and cohesion per boid
	var separations: Array[Vector2] = []
	var alignments: Array[Vector2] = []
	var cohesions: Array[Vector2] = []
	for i in range(params.boid_count):
		var separation := Vector2(0, 0)
		var alignment := Vector2(0, 0)
		var cohesion := Vector2(0, 0)
		for j in range(params.boid_count):
			var distance = (positions[i] - positions[j]).length()
			if distance < params.range * 0.5 and distance != 0:
				separation += (positions[i]-positions[j])/(distance**2)
			if distance < params.range:
				alignment += directions[j]
				cohesion += positions[j]-positions[i]
		if separation != Vector2(0, 0):
			separation /= separation.length()
		if cohesion != Vector2(0, 0):
			cohesion /= cohesion.length()
		if alignment != Vector2(0, 0):
			alignment /= alignment.length()
		separations.append(separation)
		cohesions.append(cohesion)
		alignments.append(alignment)

	for i in range(params.boid_count):
		# Apply separation, alignment and cohesion to direction vector and normalize
		directions[i] += separations[i] * params.separation_weight * delta * 2.0
		directions[i] += cohesions[i] * params.cohesion_weight * delta
		directions[i] += alignments[i] * params.alignment_weight * delta
		directions[i] /= directions[i].length()

		# Move positions in directions by delta*speed
		positions[i] += directions[i]*delta*params.speed

		# Make boid positions wrap around at borders of screen
		positions[i] = Vector2(fposmod(positions[i].x, screen_size.x), fposmod(positions[i].y, screen_size.y))
	update_boids()


func update_boids() -> void:
	"""
	Updates graphical positions and orientations of boids.
	"""
	var boids := params.get_node("Boids").get_children()
	for i in range(params.boid_count):
		var boid: Node2D = boids[i]

		# Set position of boids by updating origin of transform
		boid.transform.origin = positions[i]

		# Set rotation of boids by aligning direction with up-vector of transform
		var up := directions[i]
		var right := Vector2(up.y, -up.x)
		boid.transform.x = right*params.scale_factor
		boid.transform.y = -up*params.scale_factor
