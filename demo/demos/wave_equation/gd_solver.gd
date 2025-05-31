extends WaveSolver

var x = []
var u = []
var uprev = []
var tmp = []

func initialize() -> void:
	x = range(params.num_points).map(func(elt): return (params.dx * elt + params.xmin))
	u = params.u.duplicate()
	uprev = params.uprev.duplicate()

	tmp.resize(params.num_points)

func simulation_step(delta: float) -> void:
	var rsq = (params.wave_speed * (1/params.frame_rate/params.num_steps_per_frame) / params.dx)**2

	for n in params.num_steps_per_frame:
		for i in range(1, u.size()-1):
			tmp[i] = 2 * (1 - rsq) * u[i] - uprev[i] + (rsq * (u[i-1] + u[i+1]))

			# boundary condition
			tmp[0] = tmp[1] if (params.bc_left) else (2 * (1 - rsq) * u[0] - uprev[0] + rsq * u[1])
			tmp[-1] = tmp[-2] if (params.bc_right) else  (2 * (1 - rsq) * u[-1] - uprev[-1] + rsq * u[-2])

		uprev = u.duplicate()
		u = tmp.duplicate()

func on_draw() -> void:
	for idx in params.draw_array.size():
		params.draw_array[idx].x = params.xscale * x[params.draw_range[idx]]
		params.draw_array[idx].y = params.yscale * u[params.draw_range[idx]]

	params.draw_polyline(params.draw_array, params.point_color, params.point_size)

	if params.bc_left:
		params.draw_circle(Vector2(x[0], u[0] * params.yscale), params.anchor_size, params.anchor_color)

	if params.bc_right:
		params.draw_circle(Vector2(x[x.size() - 1] * params.xscale, u[u.size() - 1] * params.yscale), params.anchor_size, params.anchor_color)
