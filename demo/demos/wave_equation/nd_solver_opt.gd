extends Solver

var rng: Vector4i
var x: NDArray

var u: NDArray
var u_rng: NDArray
var u_from_2: NDArray
var u_to_num_points: NDArray

var uprev: NDArray
var uprev_rng: NDArray

# temporary arrays
var tmp1: NDArray
var tmp2: NDArray
	
var rsq: NDArray
var rsq_inv: NDArray

func initialize() -> void:
	# solution range
	rng = nd.range(1, params.num_points + 1)

	# grids and solution arrays
	x = nd.linspace(params.xmin, params.xmax, params.num_points)

	u = nd.zeros(params.num_points + 2)
	u_rng = u.get(rng)
	u_from_2 = u.get(nd.from(2))
	u_to_num_points = u.get(nd.to(params.num_points))

	uprev = nd.zeros(params.num_points + 2)
	uprev_rng = uprev.get(rng)
	
	# tmp arrays
	tmp1 = nd.zeros(params.num_points)
	tmp2 = nd.zeros(params.num_points)

	# initial condition
	u_rng.set(params.u)
	uprev_rng.set(params.uprev)
	
	# Other variables
	rsq = nd.array((params.wave_speed * (1/params.frame_rate/params.num_steps_per_frame) / params.dx)**2)
	rsq_inv = nd.multiply(2, nd.subtract(1, rsq))

func simulation_step(delta: float) -> void:	
	for i in params.num_steps_per_frame:
		tmp1.assign_multiply(rsq_inv, u_rng)
		tmp1.assign_subtract(tmp1, uprev_rng)

		tmp2.assign_add(u_from_2, u_to_num_points)
		tmp2.assign_multiply(rsq, tmp2)

		uprev.set(u)
		u_rng.assign_add(tmp1, tmp2)

		# boundary condition
		if params.bc_left: u.set(u.get(1), 0); uprev.set(uprev.get(1), 0)
		if params.bc_right: u.set(u.get(params.num_points), params.num_points+1); uprev.set(uprev.get(params.num_points), params.num_points+1)

func on_draw() -> void:
	for i in range(0, params.num_points, max(1, floori(params.num_points/params.num_draw_points))):
		params.draw_circle(Vector2(x.get_float(i) * params.xscale, u.get_float(i) * params.yscale), params.point_size, params.point_color)

	if params.bc_left:
		params.draw_circle(Vector2(x.get_float(0), u.get_float(0) * params.yscale), params.anchor_size, params.anchor_color)

	if params.bc_right:
		params.draw_circle(Vector2(x.get_float(x.size() - 1) * params.xscale, u.get_float(u.size() - 1) * params.yscale), params.anchor_size, params.anchor_color)
