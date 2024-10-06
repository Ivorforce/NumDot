extends SIRSolver

var grid: NDArray
var gridp: NDArray
var indices: NDArray

var susceptible_mask: NDArray
var infected_mask: NDArray
var terminal_mask: NDArray

var neighbor_indices_relative: NDArray

func initialize() -> void:
	grid = nd.zeros([params.N, params.N], nd.Int64)
	gridp = nd.zeros([params.N, params.N], nd.Int64)
	indices = nd.zeros([params.N, params.N], nd.Int64)

	susceptible_mask = nd.full([params.N, params.N], false, nd.Bool)
	infected_mask = nd.full([params.N, params.N], false, nd.Bool)
	terminal_mask = nd.full([params.N, params.N], false, nd.Bool)

	neighbor_indices_relative = nd.array([+1, -1, -params.N, +params.N], nd.Int64)
	
	# indices
	for i in params.N:
		for j in params.N:
			var idx = params.N * i + j
			indices.set(idx, i, j)

	# boundary conditions
	grid.set(params.tau0 + 1, nd.range(0, 1), null)
	grid.set(params.tau0 + 1, nd.range(params.N - 1, params.N), null)
	grid.set(params.tau0 + 1, null, nd.range(0, 1))
	grid.set(params.tau0 + 1, null, nd.range(params.N - 1, params.N))
	
	# random initial infection
	place_random()
	gridp = nd.copy(grid)

func simulation_step() -> void:
	# infect susceptible cells based on infected neighbours
	susceptible_mask.assign_equal(gridp, 0)
	var infp = indices.get(susceptible_mask).to_godot_array().map(frac_infected_neighbours)
	var to_infect_mask = nd.less(nd.default_rng(Time.get_ticks_usec()).random(infp.size()), infp)
	nd.reshape(grid, -1).set(1, indices.get(susceptible_mask).get(to_infect_mask))
	
	# increment day in infection + recovery stage
	infected_mask.assign_logical_and(nd.greater(gridp, 0), nd.less(gridp, params.tau0))
	grid.set(nd.add(gridp.get(infected_mask), 1), infected_mask)
	
	# transition from recovered to susceptible
	terminal_mask.assign_equal(gridp, params.tau0)
	grid.set(0, terminal_mask)
	
	gridp = nd.copy(grid)

func frac_infected_neighbours(idx: NDArray) -> float:
	var nbs := nd.reshape(gridp, -1).get(nd.add(idx, neighbor_indices_relative))
	var num_infected = nd.sum(nd.logical_and(nd.greater(nbs, 0), nd.less_equal(nbs, params.tauI)))
	
	if num_infected == null:
		return 0.
	else:
		return num_infected.to_float() / nbs.size()
	
func on_draw() -> void:
	for i in params.N:
		for j in params.N:
			params._image.set_pixel(i, j, params.colors[grid.get_int(i, j)])

	params.update_texture()

func place_random() -> void:
	grid.set(1, randi_range(1, params.N-2), randi_range(1, params.N-2))
