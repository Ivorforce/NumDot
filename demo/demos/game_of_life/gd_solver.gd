extends GameOfLifeSolver

var grid_front: PackedByteArray
var grid_back: PackedByteArray
var image_data: PackedByteArray

func initialize() -> void:
	grid_front = PackedByteArray()
	grid_front.resize(params.N * params.N)
	grid_back = grid_front.duplicate()
	image_data = grid_front.duplicate()
	place_random()

func to_idx(x: int, y: int) -> int:
	return x + params.N * y

func simulation_step() -> void:
	for i in range(1, params.N-1):
		for j in range(1, params.N-1):
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
	for i in image_data.size():
		image_data[i] = grid_front[i] * 255
	params._image.set_data(params.N, params.N, false, Image.FORMAT_R8, image_data)
	params.update_texture()

func place_random() -> void:
	for x in range(params.N - 2):
		for y in range(params.N - 2):
			grid_front[to_idx(x + 1, y + 1)] = randi() % 2
