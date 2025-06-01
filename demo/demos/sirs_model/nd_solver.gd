extends SIRSolver

var grid_time_since_infection: NDArray
var grid_can_infect_neighbor: NDArray
var grid_is_infectable: NDArray
var grid_infected_neighbor_count: NDArray
var grid_infected_neighbor_count_inner: NDArray
var grid_infected_neighbor_ratio: NDArray
var grid_new_infected_this_step: NDArray

var neighbor_kernel: NDArray
var rng := nd.default_rng()

func initialize() -> void:
	var grid_size := [params.N, params.N]
	
	grid_time_since_infection = nd.zeros(grid_size, nd.Int64)
	grid_time_since_infection.set(params.tau0 + 1)
	grid_can_infect_neighbor = nd.zeros(grid_size, nd.Bool)
	grid_is_infectable = nd.zeros(grid_size, nd.Bool)
	grid_infected_neighbor_count = nd.zeros(grid_size, nd.Int64)
	grid_infected_neighbor_count_inner = grid_infected_neighbor_count.get(nd.range(1, -1), nd.range(1, -1))
	grid_infected_neighbor_ratio = nd.zeros(grid_size, nd.Float64)
	grid_new_infected_this_step = nd.zeros(grid_size, nd.Bool)
	
	neighbor_kernel = nd.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], nd.Int64);
	
	# random initial infection
	place_random()
	set_border_color()
	
func simulation_step() -> void:
	# Infect
	grid_can_infect_neighbor.assign_less_equal(grid_time_since_infection, params.tauI)
	grid_is_infectable.assign_greater(grid_time_since_infection, params.tau0)
	
	grid_infected_neighbor_count_inner.assign_convolve(grid_can_infect_neighbor, neighbor_kernel)
	grid_infected_neighbor_ratio.assign_divide(grid_infected_neighbor_count, 4.0)
	
	grid_infected_neighbor_ratio.assign_multiply(grid_infected_neighbor_ratio, params.spread)
	grid_new_infected_this_step.assign_less(rng.random(grid_time_since_infection.shape()), grid_infected_neighbor_ratio)
	grid_new_infected_this_step.assign_logical_and(grid_new_infected_this_step, grid_is_infectable)
	
	grid_time_since_infection.set(0, grid_new_infected_this_step)
	
	# Time Pass
	grid_time_since_infection.assign_add(grid_time_since_infection, 1)
	# clamp value (not strictly necessary given Int64, but just for completeness)
	grid_time_since_infection.assign_minimum(grid_time_since_infection, params.tau0 + 1)
	
func on_draw() -> void:
	for i in range(1, params.N-1):
		for j in range(1, params.N-1):
			var color_idx = grid_time_since_infection.get_int(i, j)
			if color_idx > params.tau0:
				color_idx = 0
			params._image.set_pixel(i, j, params.colors[color_idx])

	params.update_texture()

func place_random() -> void:
	grid_time_since_infection.set(0, randi_range(1, params.N-2), randi_range(1, params.N-2))

func set_border_color() -> void:
	for i in params.N:
		params._image.set_pixel(i, params.N-1, Color.BLACK)
		params._image.set_pixel(i, 0, Color.BLACK)
		params._image.set_pixel(params.N-1, i, Color.BLACK)
		params._image.set_pixel(0, i, Color.BLACK)

	pass
