extends KuramotoSolver

var omega: NDArray # natural frequencies
var phase: NDArray # oscillator phases

# pre-allocations
var rng := nd.default_rng()

var phase_coherence: NDArray
var avg_phase: NDArray
var phase_sin: NDArray
var phase_cos: NDArray

# rk4
var k1: NDArray
var k2: NDArray
var k3: NDArray
var k4: NDArray

func initialize() -> void:
	integrator = [Callable(euler_step), Callable(rk4_step)]

	phase = nd.multiply(2*PI, rng.random(params.N))
	omega = nd.add(nd.multiply(rng.randn(params.N), params.frequency_sigma), params.frequency_mean)
	
	phase_coherence = nd.zeros(params.N)
	avg_phase = nd.zeros(params.N)
	phase_sin = nd.zeros(params.N)
	phase_cos = nd.zeros(params.N)
	
	k1 = nd.zeros(params.N)
	k2 = nd.zeros(params.N)
	k3 = nd.zeros(params.N)
	k4 = nd.zeros(params.N)

func simulation_step() -> void:
	var dt_substep: float = params.dt / params.sub_steps
	for i in params.sub_steps: integrator[params.integrator_idx].call(dt_substep)

func compute_derivative(df: NDArray, phase: NDArray):
	phase_sin.assign_sin(phase)
	phase_cos.assign_cos(phase)
	phase_coherence.assign_divide(nd.sqrt(nd.add(nd.square(nd.sum(phase_cos)), nd.square(nd.sum(phase_sin)))), phase.size())
	avg_phase.assign_atan2(nd.sum(phase_sin), nd.sum(phase_cos))
	
	df.assign_subtract(omega, nd.multiply(nd.multiply(params.coupling, phase_coherence), nd.sin(nd.subtract(phase, avg_phase))))

func euler_step(dt: float) -> void:
	compute_derivative(k1, phase)
	phase.assign_add(phase, nd.multiply(k1, dt))

func rk4_step(dt: float) -> void:
	compute_derivative(k1, phase)
	compute_derivative(k2, nd.add(phase, nd.multiply(k1, dt/2)))
	compute_derivative(k3, nd.add(phase, nd.multiply(k2, dt/2)))
	compute_derivative(k4, nd.add(phase, nd.multiply(k3, dt)))
	
	phase.assign_add(phase, nd.multiply(dt/6, nd.add(k1, nd.add(nd.multiply(2, k2), nd.add(nd.multiply(2, k3), k4)))))

func on_draw() -> void:
	for i in params.N:
		params.draw_circle(params.positions[i], params.entity_size, Color(params.entity_color, sin(phase.get_float(i))/2 + 0.5))

func generate_frequencies() -> void:
	omega = nd.add(nd.multiply(rng.randn(params.N), params.frequency_sigma), params.frequency_mean)
