extends Solver

var x = null
var u  = null
var uprev = null
var rng = null

# temporary arrays
var tmp1 = null
var tmp2 = null
	
func initialize() -> void:
	# grids and solution arrays
	x = nd.linspace(params.xmin, params.xmax, params.num_points)
	u = nd.zeros(params.num_points + 2)
	uprev = nd.zeros(params.num_points + 2)
	
	# tmp arrays
	tmp1 = nd.zeros(params.num_points)
	tmp2 = nd.zeros(params.num_points)

	# solution range
	rng = nd.range(1, params.num_points + 1)
	
	# initial condition
	u.get(rng).assign_multiply(nd.exp(nd.subtract(0, nd.divide(nd.square(nd.subtract(x, 0.5)), 0.005))), -2)
	uprev = nd.array(u)

func simulation_step(delta: float) -> void:
	var rsq = (params.wave_speed * (1/params.frame_rate/params.num_steps_per_frame) / params.dx)**2
	
	for i in params.num_steps_per_frame:
		tmp1.assign_multiply(2 * (1 - rsq), u.get(rng))
		tmp1.assign_subtract(tmp1, uprev.get(rng))

		tmp2.assign_add(u.get(nd.from(2)), u.get(nd.to(params.num_points)))
		tmp2.assign_multiply(rsq, tmp2)

		uprev = nd.array(u) # copy
		u.get(rng).assign_add(tmp1, tmp2)

func on_draw() -> void:
	for i in range(0, params.num_points, max(1, floori(params.num_points/params.num_draw_points))):
		params.draw_circle(Vector2(x.get_float(i) * params.xscale, u.get_float(i) * params.yscale), 2., Color.RED)
