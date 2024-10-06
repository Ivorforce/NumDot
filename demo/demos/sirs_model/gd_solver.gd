extends SIRSolver

@onready var grid: Array
@onready var gridp: Array

func initialize() -> void:
	grid = create_matrix(params.N)

	# boundary conditions
	grid[0].fill(params.tau0 + 1)
	grid[-1].fill(params.tau0 + 1)
	
	for arr in grid:
		arr[0] = params.tau0 + 1 
		arr[-1] = params.tau0 + 1

	place_random()
	gridp = grid.duplicate(true)

func simulation_step() -> void:
	var infp := 0. 
	
	for i in range(1, params.N-1):
		for j in range(1, params.N-1):
			if gridp[i][j] == 0:
				infp = [gridp[i+1][j], gridp[i-1][j], gridp[i][j-1], gridp[i][j+1]].map(func(elt): return (elt > 0) and (elt <= params.tauI)).count(true) / 4.
				if randf() < infp:
					grid[i][j] = 1
			elif gridp[i][j] < params.tau0:
				grid[i][j] += 1
			elif gridp[i][j] == params.tau0:
				grid[i][j] = 0

	gridp = grid.duplicate(true)

func on_draw() -> void:
	for i in params.N:
		for j in params.N:
			params._image.set_pixel(i, j, params.colors[grid[i][j]])

	params.update_texture()
	
func create_matrix(N: int) -> Array:
	var matrix := []
	for x in range(N):
		matrix.append([])
		for y in range(N):
			matrix[x].append(0)

	return matrix

func place_random() -> void:
	grid[randi_range(1, params.N-2)][randi_range(1, params.N-2)] = 1
