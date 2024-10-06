extends Node2D

@export var num_points: int = 150

@export_category("Simulation parameters")
@export var xmin: float = 0.
@export var xmax: float = 1.
@export var wave_speed: float = 0.25
@export var frame_rate: float = 60

@export_category("Visualization parameters")
@export var xscale: float = 500.
@export var yscale: float = 50.
@export var num_draw_points: int = 200
@export var point_color := Color("70d6ff")
@export var point_size := 2.
@export var anchor_size := 4.
@export var anchor_color := Color("000000")
@export var wall_width := 3

@export_category("Solver parameters")
@export var solver: WaveSolver

# simulation parameters computed
@onready var dx: float = (xmax - xmin)/num_points
@onready var num_steps_per_frame: int = max(1, ceili(wave_speed/ dx / frame_rate)) # to satisfy CFL condition
	
@export var init_params = [
	{"x0": 0.5, "sig": 0.005, "amplitude": -2.},
	{"mode": 5, "amplitude": 1.},
	{"xi": 0.2},
	{"x1": 0.2, "x2": 0.8}
]

var init_option = 0

# initial arrays
var x = []
var u = []
var uprev = []

@export_category("Boundary conditions")
@export var bc_left = 0
@export var bc_right = 0

# fps
var frame_time: float = 1./60

# drawing arrays for the wave
var draw_range: Array # range of solution points to draw
var draw_array: PackedVector2Array # final array to pass to draw_polyline

func _ready() -> void:
	_on_init_option_item_selected(init_option)

func _process(delta: float) -> void:
	%FPSLabel.text = "FPS: " + str(int(1/delta))
	%IterLabel.text = "Sub-steps: " + str(num_steps_per_frame)

	frame_time = Time.get_ticks_usec()
	solver.simulation_step(delta)
	frame_time = Time.get_ticks_usec() - frame_time
	
	%FrameTimeLabel.text = "Delta (ms): " + str(snappedf(frame_time/1000, 1e-3))
	queue_redraw()

func _draw() -> void:
	if not bc_left: draw_line(Vector2(-3, -0.5 * yscale), Vector2(-3, 0.5 * yscale), Color.WHITE, wall_width)
	if not bc_right: draw_line(Vector2(xmax * xscale, -0.5 * yscale), Vector2(xmax * xscale, 0.5 * yscale), Color.WHITE, wall_width)
	solver.on_draw()

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	restart_simulation()
	
func _on_point_slider_drag_ended(value_changed: bool) -> void:
	%PointLabel.text = "Points: " + str(%PointSlider.value)
	%CFLLabel.text = "CFL: " + str(snappedf(wave_speed * 1/(frame_rate * num_steps_per_frame * dx), 1e-3))
	
	num_points = %PointSlider.value
	dx = (xmax - xmin)/num_points
	num_steps_per_frame = max(1, ceili(wave_speed/ dx / frame_rate)) # to satisfy CFL condition
	set_initial_condition(init_option)
	restart_simulation()

func _on_init_option_item_selected(index: int) -> void:
	init_option = index
	set_initial_condition(init_option)
	restart_simulation()

func set_initial_condition(idx) -> void:
	x = range(num_points).map(func(elt): return (dx * elt + xmin))
	
	u.resize(num_points)
	uprev.resize(num_points)
	
	match idx:
		0:
			for i in u.size():
				u[i] = init_params[0]["amplitude"] * exp(-(x[i] - init_params[0]["x0"])**2/init_params[0]["sig"])
				uprev[i] = u[i]
		1:
			for i in u.size():
				u[i] = init_params[1]["amplitude"] * sin(init_params[1]["mode"] * PI * x[i])
				uprev[i] = u[i]
		2: 
			for i in u.size():
				u[i] = init_params[0]["amplitude"] * exp(-(x[i] - init_params[2]["xi"])**2/init_params[0]["sig"])
				var delx = wave_speed/frame_rate/num_steps_per_frame
				uprev[i] = init_params[0]["amplitude"] * exp(-(x[i] - init_params[2]["xi"] - delx)**2/init_params[0]["sig"])
		3:
			for i in u.size():
				u[i] = init_params[0]["amplitude"] * exp(-(x[i] - init_params[3]["x1"])**2/init_params[0]["sig"]) - init_params[0]["amplitude"] * exp(-(x[i] - init_params[3]["x2"])**2/init_params[0]["sig"])
				
				var delx = wave_speed/frame_rate/num_steps_per_frame
				uprev[i] = init_params[0]["amplitude"] * exp(-(x[i] - init_params[3]["x1"] - delx)**2/init_params[0]["sig"]) - init_params[0]["amplitude"] * exp(-(x[i] - init_params[3]["x2"] + delx)**2/init_params[0]["sig"])

func _on_bc_left_item_selected(index: int) -> void:
	bc_left = index

func _on_bc_right_item_selected(index: int) -> void:
	bc_right = index

func restart_simulation() -> void:
	var draw_step = max(1, floori(num_points/num_draw_points))
	draw_range = range(0, num_points, draw_step)
	draw_array.resize(draw_range.size())
	draw_array.fill(Vector2.ZERO)

	solver.initialize()
