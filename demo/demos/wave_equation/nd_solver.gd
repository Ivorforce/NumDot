extends WaveSolver

var x = null
var u  = null
var uprev = null
var tmp = null
var rng = null

func initialize() -> void:
	# grids and solution arrays
	x = nd.linspace(params.xmin, params.xmax, params.num_points)
	u = nd.zeros(params.num_points + 2)
	uprev = nd.zeros(params.num_points + 2)
	tmp = nd.zeros(params.num_points + 2)
	
	# solution range
	rng = nd.range(1, params.num_points + 1)
	
	# initial condition
	u.set(nd.array(params.u), rng)
	uprev.set(nd.array(params.uprev), rng)

func simulation_step(delta: float) -> void:
	var rsq = (params.wave_speed * (1/params.frame_rate/params.num_steps_per_frame) / params.dx)**2
	
	for i in params.num_steps_per_frame:
		tmp.set(
			nd.add(
				nd.subtract(nd.multiply(2 * (1 - rsq), u.get(rng)), uprev.get(rng)), 
				nd.multiply(rsq, nd.add(u.get(nd.from(2)), u.get(nd.to(params.num_points))))
			), rng)
		
		# boundary condition
		if params.bc_left: tmp.set(tmp.get(1), 0)
		if params.bc_right: tmp.set(tmp.get(params.num_points), params.num_points+1)

		uprev = u
		u = nd.array(tmp) # copy

func on_draw() -> void:
	for i in range(0, params.num_points, max(1, floori(params.num_points/params.num_draw_points))):
		params.draw_circle(Vector2(x.get_float(i) * params.xscale, u.get_float(i) * params.yscale), params.point_size, params.point_color)

	if params.bc_left:
		params.draw_circle(Vector2(x.get_float(0), u.get_float(0) * params.yscale), params.anchor_size, params.anchor_color)

	if params.bc_right:
		params.draw_circle(Vector2(x.get_float(x.size() - 1) * params.xscale, u.get_float(u.size() - 1) * params.yscale), params.anchor_size, params.anchor_color)
