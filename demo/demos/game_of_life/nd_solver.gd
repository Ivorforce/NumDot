extends GameOfLifeSolver

var is_alive: NDArray
var is_alive_inner: NDArray
var neighour_count: NDArray
var neighbor_kernel: NDArray
var tmp_inner: NDArray

var rng := nd.default_rng()

func initialize() -> void:
	var grid_size := [params.N, params.N]
	
	# TODO This should be bool, but that's somehow broken with the assignment
	#  from random below
	is_alive = nd.zeros(grid_size, nd.Int8)
	is_alive_inner = is_alive.get(nd.range(1, -1), nd.range(1, -1))

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
	for i in range(params.N):
		for j in range(params.N):
			params._image.set_pixel(i, j, params.color_on if is_alive.get_bool(i, j) else params.color_off)

	params.update_texture()

func place_random() -> void:
	is_alive_inner.set(rng.integers(0, 2, is_alive_inner.shape(), nd.Int8))
