extends GameOfLifeSolver

@onready var grid_front: Array[PackedByteArray]
@onready var grid_back: Array[PackedByteArray]

func initialize() -> void:
	grid_front = create_matrix(params.N)

	place_random()
	grid_back = grid_front.duplicate(true)

func simulation_step() -> void:
	for i in range(1, params.N-1):
		for j in range(1, params.N-1):
			var neighbor_count := grid_front[i - 1][j - 1] + grid_front[i][j - 1] + grid_front[i + 1][j - 1]\
				+ grid_front[i - 1][j] + grid_front[i + 1][j]\
				+ grid_front[i - 1][j + 1] + grid_front[i][j + 1] + grid_front[i + 1][j + 1]

			if grid_front[i][j]:
				grid_back[i][j] = 1 if (neighbor_count >= 2 and neighbor_count <= 3) else 0
			else:
				grid_back[i][j] = 1 if (neighbor_count == 3) else 0
	
	# swap front / back
	var tmp := grid_front
	grid_front = grid_back
	grid_back = tmp

func on_draw() -> void:
	for i in params.N:
		for j in params.N:
			params._image.set_pixel(i, j, params.color_on if grid_front[i][j] else params.color_off)

	params.update_texture()
	
func create_matrix(N: int) -> Array[PackedByteArray]:
	var matrix: Array[PackedByteArray] = []
	for x in range(N):
		var row := PackedByteArray()
		row.resize(N)
		matrix.append(row)

	return matrix

func place_random() -> void:
	for x in range(grid_front.size() - 2):
		var row := grid_front[x + 1]
		for y in range(row.size() - 2):
			row[y + 1] = randi() % 2
