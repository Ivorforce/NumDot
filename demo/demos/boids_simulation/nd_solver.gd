extends BoidsSolver

var positions: NDArray
var directions: NDArray

var rng := nd.default_rng()

var screen_size: Vector2

func initialize() -> void:
	"""
	Initializes solver by setting the screen size
	and generating initial position and direction arrays.
	"""
	screen_size = params.get_viewport_rect().size

	# Initialize position and direction vector
	positions = initialize_position_array(params.boid_count)
	directions = initialize_direction_array(params.boid_count)


# Helper function to create position direction vector with length
func initialize_position_array(length: int) -> NDArray:
	"""
	Generates an NDArray of random positions within screen bounds.

	Parameters:
	length (int): Number of positions to generate.

	Returns:
	NDArray: Array containing random position vectors.
	"""
	# Initialize position vector of shape [length, 2]
	# Values are random 2D positions on screen
	var positions_xy = rng.random([length, 2])
	positions_xy.assign_multiply(positions_xy, nd.array([screen_size.x, screen_size.y], 2))
	return positions_xy


# Helper function to create random direction vector with length
func initialize_direction_array(length: int) -> NDArray:
	"""
	Creates an NDArray of normalized random direction vectors.

	Parameters:
	length (int): Number of directions to generate.

	Returns:
	NDArray: Array containing random normalized direction vectors.
	"""
	# Initialize angle vector of shape [length]
	# Values are random angles in [0, 2*PI), used to create direction vector
	var angles := rng.random([length])
	angles.assign_multiply(angles, 2.0 * PI)

	# Initialize direction vector of shape [length, 2]
	# Values are normalized 2D direction vectors according to angles
	var directions_x := nd.cos(angles)
	var directions_y := nd.sin(angles)
	return nd.stack([directions_x, directions_y], 1)


func simulation_step(delta: float) -> void:
	"""
	Executes a simulation step updating positions and directions
	using NumDot operations for efficiency.

	Parameters:
	delta (float): Time elapsed since the last frame.
	"""
	# Check if boid_count has been changed, update vector sizes accordingly
	var boid_count_difference = params.boid_count-positions.shape()[0]
	if boid_count_difference < 0:
		positions = positions.get(nd.range(params.boid_count), nd.range(2))
		directions = directions.get(nd.range(params.boid_count), nd.range(2))
	elif boid_count_difference > 0:
		var new_positions := initialize_position_array(boid_count_difference)
		var new_directions := initialize_direction_array(boid_count_difference)
		positions = nd.vstack([positions, new_positions])
		directions = nd.vstack([directions, new_directions])

	# Move positions in directions by delta*speed
	var offset := nd.multiply(directions, delta*params.speed)
	positions.assign_add(positions, offset)

	# Make boid positions wrap around at borders of screen
	for axis in [0, 1]:
		var positions_axis := positions.get(nd.range(params.boid_count), axis)
		var wrap_positive := nd.greater(positions_axis, screen_size[axis]).as_type(nd.Int16)
		var wrap_negative := nd.less(positions_axis, 0).as_type(nd.Int16)
		wrap_positive.assign_multiply(wrap_positive, -screen_size[axis])
		wrap_negative.assign_multiply(wrap_negative, screen_size[axis])
		positions_axis.assign_add(positions_axis, wrap_positive)
		positions_axis.assign_add(positions_axis, wrap_negative)
		positions.set(positions_axis, nd.range(params.boid_count), axis)

	# Create pair-wise position differences of boids with shape [n, n, 2]
	var position_differences := nd.subtract(positions.get(null, &"newaxis", null), positions.get(&"newaxis", null, null))
	# Calculate distances from position differences with shape [n, n]
	var position_distances := nd.norm(position_differences, 2, 2)
	# Mark every pair of boids with distance smaller than range in separation mask with shape [n, n]
	var vision_mask := nd.less_equal(position_distances, params.range).as_type(nd.Int16)
	# Mark every pair of boids with distance smaller than 0.5*range in separation mask with shape [n, n]
	var separation_mask := nd.less_equal(position_distances, params.range*0.5).as_type(nd.Int16)

	# Separation
	# Calculate separation direction normalization divisor with shape [n, n]
	var separation_normalization := nd.square(position_distances)
	separation_normalization.assign_add(separation_normalization, nd.eye(params.boid_count))
	# Normalize separation directions inversely proportional to distances with shape [n, n, 1]
	var separation_directions := nd.divide(position_differences, separation_normalization.get(null, null, &"newaxis"))
	# Ignore boids not marked in separation mask
	separation_directions.assign_multiply(position_differences, separation_mask.get(null, null, &"newaxis"))
	# Calculate sum of separation directions per boid with shape [n, 2]
	var separations := nd.sum(separation_directions, 0)
	# Make seperation directions point away from boids in separation range
	separations.assign_multiply(separations, -1)
	# Calculate separation direction normalization divisor with shape [n]
	var separations_normalization := nd.norm(separations, 2, 1)
	separations_normalization.assign_add(separations_normalization, nd.equal(separations_normalization, 0.0).as_type(nd.Int16))
	# Normalize separation directions for each boid
	separations.assign_divide(separations, separations_normalization.get(null, &"newaxis"))

	# Alignment
	# Ignore boids not marked in vision mask
	var alignment_directions := nd.multiply(directions.get(null, &"newaxis", null), vision_mask.get(null, null, &"newaxis"))
	# Calculate sum of alignment directions per boid with shape [n, 2]
	var alignments := nd.sum(alignment_directions, 0)
	# Normalize alignment directions for each boid
	alignments.assign_divide(alignments, nd.norm(alignments, 2, 1).get(null, &"newaxis"))

	# Cohesion
	# Ignore boids not marked in cohesion mask
	var cohesion_positions := nd.multiply(positions.get(null, &"newaxis", null), vision_mask.get(null, null, &"newaxis"))
	# Calculate sum of cohesion positions with shape [n, 1]
	var cohesions := nd.sum(cohesion_positions, 0)
	# Find cohesion centers by calculating averages
	cohesions.assign_divide(cohesions, nd.sum(vision_mask, 0).get(null, &"newaxis"))
	# Calculate cohesion directions by taking difference between boids and respective cohesion centers
	cohesions.assign_subtract(cohesions, positions)
	# Calculate cohesion direction normalization divisor (dist to cohesion center if existing, else 1)
	var cohesions_normalization = nd.norm(cohesions, 2, 1)
	cohesions_normalization.assign_add(cohesions_normalization, nd.equal(cohesions_normalization, 0.0).as_type(nd.Int16))
	# Normalize cohesion directions
	cohesions.assign_divide(cohesions, cohesions_normalization.get(null, &"newaxis"))

	# Update directions vector according to separation, alignment and cohesion with respective weights
	directions.assign_add(directions, separations.assign_multiply(separations, params.separation_weight*delta*2))
	directions.assign_add(directions, alignments.assign_multiply(alignments, params.alignment_weight*delta))
	directions.assign_add(directions, cohesions.assign_multiply(cohesions, params.cohesion_weight*delta))

	# Normalize direction vection lengths to 1
	directions.assign_divide(directions, nd.norm(directions, 2, 1).get(null, &"newaxis"))

	update_boids()


func update_boids() -> void:
	"""
	Updates the graphical representations of boids
	to match the computed positions and directions.
	"""
	var boids := params.get_node("Boids").get_children()
	for i in range(min(params.boid_count, boids.size())):
		var boid: Node2D = boids[i]

		# Set position of boids by updating origin of transform
		boid.transform.origin = positions.get_vector2(i, nd.range(2))

		# Set rotation of boids by aligning direction with up-vector of transform
		var up := directions.get_vector2(i, nd.range(2))
		var right := Vector2(up.y, -up.x)
		boid.transform.x = right*params.scale_factor
		boid.transform.y = -up*params.scale_factor
