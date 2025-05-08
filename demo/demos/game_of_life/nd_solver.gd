extends GameOfLifeSolver

var is_alive: NDArray
var is_alive_inner: NDArray
var image_data: NDArray
var neighour_count: NDArray
var neighbor_kernel: NDArray
var tmp_inner: NDArray

var rng := nd.default_rng()

func initialize() -> void:
	var grid_size := [params.N + 2, params.N + 2]

	is_alive = nd.zeros(grid_size, nd.Bool)
	is_alive_inner = is_alive.get(nd.range(1, -1), nd.range(1, -1))
	image_data = nd.empty_like(is_alive_inner)

	neighour_count = nd.zeros_like(is_alive_inner, nd.Int8)
	tmp_inner = nd.empty_like(is_alive_inner, nd.Bool)

	neighbor_kernel = nd.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], nd.Int8);

	place_random()

func simulation_step() -> void:
	neighour_count.assign_convolve(is_alive, neighbor_kernel)

	# This uses masks, not sure if that's the best performing action here.
	# Boolean operations may be accelerated better!
	is_alive_inner.set(false, tmp_inner.assign_less(neighour_count, 2))
	is_alive_inner.set(true, tmp_inner.assign_equal(neighour_count, 3))
	is_alive_inner.set(false, tmp_inner.assign_greater(neighour_count, 3))

func on_draw() -> void:
	image_data.set(is_alive_inner)
	var image_data := nd.reshape(image_data, -1).to_packed_byte_array()
	params._image.set_data(params.N, params.N, false, Image.FORMAT_R8, image_data)
	params.update_texture()

func place_random() -> void:
	is_alive_inner.set(rng.integers(0, 2, is_alive_inner.shape(), nd.Bool))
