extends BoidsSolver

var positions: NDArray
var directions: NDArray
var noise_positions: NDArray

var rng := nd.default_rng()
var screen_size: Vector2

func initialize() -> void:
	screen_size = params.get_viewport_rect().size

	# Initialize position and direction vector
	positions = initialize_position_array(params.boid_count)
	directions = initialize_direction_array(params.boid_count)

	# Initalize vector with random noise sampling positions
	noise_positions = initialize_sample_position_array(params.boid_count)


# Helper function to create position direction vector with length
func initialize_position_array(length: int) -> NDArray:
	# Initialize position vector of shape [length, 2]
	# Values are random 2D positions on screen
	var positions_xy = rng.random([length, 2])
	positions_xy.assign_multiply(positions_xy, nd.array([screen_size.x, screen_size.y], 2))
	return positions_xy


# Helper function to create random direction vector with length
func initialize_direction_array(length: int) -> NDArray:
	# Initialize angle vector of shape [length]
	# Values are random angles in [0, 2*PI), used to create direction vector
	var angles := rng.random([length])
	angles.assign_multiply(angles, 2.0 * PI)

	# Initialize direction vector of shape [length, 2]
	# Values are normalized 2D direction vectors according to angles
	var directions_x := nd.cos(angles)
	var directions_y := nd.sin(angles)
	return nd.stack([directions_x, directions_y], 1)


# Helper function to create random sample position vector with length
func initialize_sample_position_array(length: int) -> NDArray:
	# Initialize noise sampling position vector of shape [length]
	return nd.add(rng.integers(1e3, null, [length]), rng.random([length]))


func simulation_step(delta: float) -> void:
	# Check if boid_count has been changed, update vector sizes accordingly
	var boid_count_difference = params.boid_count-positions.shape()[0]
	if boid_count_difference < 0:
		positions = positions.get(nd.range(params.boid_count), nd.range(2))
		directions = directions.get(nd.range(params.boid_count), nd.range(2))
		noise_positions = noise_positions.get(nd.range(params.boid_count))
	elif boid_count_difference > 0:
		var new_positions := initialize_position_array(boid_count_difference)
		var new_directions := initialize_direction_array(boid_count_difference)
		var new_noise_positions := initialize_sample_position_array(boid_count_difference)
		positions = nd.vstack([positions, new_positions])
		directions = nd.vstack([directions, new_directions])
		noise_positions = nd.concatenate([noise_positions, new_noise_positions])

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

	# For each force:
		# Calculate masks for boids in range
		# Calulcate difference to current boid
		# Sum over differences and normalize

	# TODO Separation
	# TODO Alignment
	# TODO Cohesion

	# TODO Noise

	# TODO Add to direction according to weigths and normalize

	update_boids()

func update_boids() -> void:
	var boids := params.get_node("Boids").get_children()
	for i in range(boids.size()):
		var boid: Node2D = boids[i]

		# Set position of boids by updating origin of transform
		boid.transform.origin = positions.get_vector2(i, nd.range(2))

		# Set rotation of boids by aligning direction with up-vector of transform
		var up := directions.get_vector2(i, nd.range(2))
		var right := Vector2(up.y, -up.x)
		boid.transform.x = right*params.scale_factor
		boid.transform.y = -up*params.scale_factor
