extends GameOfLifeSolver

var grid_front: PackedByteArray
var grid_back: PackedByteArray
var image_data: PackedByteArray

func initialize() -> void:
	grid_front = PackedByteArray()

	# Need one row / column on each side extra so the lookup later is easier
	grid_front.resize((params.N + 2) * (params.N + 2))
	grid_back = grid_front.duplicate()

	image_data = PackedByteArray()
	image_data.resize(params.N * params.N)
	place_random()

func to_idx(x: int, y: int) -> int:
	return x + (params.N + 2) * y

func simulation_step() -> void:
	for i in range(1, params.N + 1):
		for j in range(1, params.N + 1):
			var neighbor_count := grid_front[to_idx(i - 1, j - 1)] + grid_front[to_idx(i, j - 1)] + grid_front[to_idx(i + 1, j - 1)]\
				+ grid_front[to_idx(i - 1, j)] + grid_front[to_idx(i + 1, j)]\
				+ grid_front[to_idx(i - 1, j + 1)] + grid_front[to_idx(i, j + 1)] + grid_front[to_idx(i + 1, j + 1)]

			if grid_front[to_idx(i, j)]:
				grid_back[to_idx(i, j)] = 1 if (neighbor_count >= 2 and neighbor_count <= 3) else 0
			else:
				grid_back[to_idx(i, j)] = 1 if (neighbor_count == 3) else 0

	# swap front / back
	var tmp := grid_front
	grid_front = grid_back
	grid_back = tmp

func on_draw() -> void:
	for i in range(0, params.N):
		for j in range(0, params.N):
			image_data[i + params.N * j] = grid_front[to_idx(i + 1, j + 1)]

	params._image.set_data(params.N, params.N, false, Image.FORMAT_R8, image_data)
	params.update_texture()

func place_random() -> void:
	for i in range(1, params.N + 1):
		for j in range(1, params.N + 1):
			grid_front[to_idx(i, j)] = randi() % 2
