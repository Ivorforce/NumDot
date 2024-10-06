extends KuramotoSolver

var omega: Array # natural frequencies
var phase: Array # oscillator phases

# pre-allocations
var phase_coherence: float
var avg_phase: float
var phase_sin: Array
var phase_cos: Array
var phase_sin_sum: float
var phase_cos_sum: float

# rk4
var kf: Array
var k1: Array
var k2: Array
var k3: Array
var k4: Array

func initialize() -> void:
	integrator = [Callable(euler_step), Callable(rk4_step)]
	
	phase.resize(params.N)
	for i in phase.size(): phase[i] = randf_range(0, 2*PI)
		
	omega.resize(params.N)
	for i in omega.size(): omega[i] = randfn(params.frequency_mean, params.frequency_sigma)

	phase_sin.resize(params.N)
	phase_cos.resize(params.N)
	
	kf.resize(params.N)
	k1.resize(params.N)
	k2.resize(params.N)
	k3.resize(params.N)
	k4.resize(params.N)
	
func on_draw() -> void:
	for i in params.N:
		params.draw_circle(params.positions[i], params.entity_size, Color(params.entity_color, sin(phase[i])/2 + 0.5))

func simulation_step() -> void:
	var dt_substep: float = params.dt / params.sub_steps
	for i in params.sub_steps: integrator[params.integrator_idx].call(dt_substep)

func compute_derivative(df: Array, phase: Array):
	phase_sin_sum = 0.
	phase_cos_sum = 0.
	
	for i in phase.size(): 
		phase_sin[i] = sin(phase[i])
		phase_cos[i] = cos(phase[i])
		phase_sin_sum += phase_sin[i]
		phase_cos_sum += phase_cos[i]
		
	phase_coherence = sqrt(phase_sin_sum**2 + phase_cos_sum**2) / params.N
	avg_phase = atan2(phase_sin_sum, phase_cos_sum)
	
	for i in phase.size():
		df[i] = omega[i] + params.coupling * phase_coherence * sin(phase[i] - avg_phase)
	
func generate_frequencies() -> void:
	omega.resize(params.N)
	for i in omega.size(): randfn(params.frequency_mean, params.frequency_sigma)

func euler_step(dt: float) -> void:
	compute_derivative(k1, phase)
	for i in phase.size(): phase[i] += k1[i] * dt

func rk4_step(dt: float) -> void:
	compute_derivative(k1, phase)
	
	for i in phase.size(): kf[i] = phase[i] + k1[i] * dt/2
	compute_derivative(k2, kf)
	
	for i in phase.size(): kf[i] = phase[i] + k2[i] * dt/2
	compute_derivative(k3, kf)

	for i in phase.size(): kf[i] = phase[i] + k3[i] * dt
	compute_derivative(k4, kf)
	
	for i in phase.size(): phase[i] += dt/6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
