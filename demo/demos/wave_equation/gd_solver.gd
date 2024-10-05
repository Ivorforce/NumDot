extends Solver

var x = []
var u = []
var uprev = []
var tmp = []

func initialize() -> void:
	x = range(params.num_points).map(func(elt): return (params.dx * elt + params.xmin))
	u.resize(params.num_points)
	uprev.resize(params.num_points)
	tmp.resize(params.num_points)
	
	for i in u.size():
		u[i] = -2 * exp(-(x[i] - 0.5)**2/0.005)
		uprev[i] = u[i]
	
func simulation_step(delta: float) -> void:
	var rsq = (params.wave_speed * (1/params.frame_rate/params.num_steps_per_frame) / params.dx)**2

	for n in params.num_steps_per_frame:
		for i in range(1, u.size()-1):
			tmp[i] = 2 * (1 - rsq) * u[i] - uprev[i] + (rsq * (u[i-1] + u[i+1]))
		
			tmp[0] = 2 * (1 - rsq) * u[0] - uprev[0] + rsq * u[1]
			tmp[-1] = 2 * (1 - rsq) * u[-1] - uprev[-1] + rsq * u[-2]

		uprev = u.duplicate()
		u = tmp.duplicate()	

func on_draw() -> void:
	for i in range(0, params.num_points, max(1, floori(params.num_points/params.num_draw_points))):
		params.draw_circle(Vector2(x[i] * params.xscale, u[i] * params.yscale), 2., Color.RED)
