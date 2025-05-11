extends Node2D


@export_category("Simulation parameters")
@export var solver: BoidsSolver






@export var N: int = 20
# Implement slider functionalities for:
	# Number of Boids
	# Speed
	# Visual range
	# Seperation
	# Alignment
	# Cohesion

# Implement reset

func _ready() -> void:
	
	# Set correct values in GUI

	# Initialize boids as scenes from res://demos/boids_simulation/boid.tscn
	# Add border around screen redirecting boids to screen center
	

	#TODO fix workaround below 
	self.solver = $Solvers/GDSolver
	solver.initialize()
	pass

func _process(delta: float) -> void:
	# Call simulation_step(delta)
	solver.simulation_step(delta)
	# Update GUI

	pass

func _on_point_slider_drag_ended(value_changed: bool) -> void:
	N = %PointSlider.value
	%PointLabel.text = "Grid: " + str(N) + "x" + str(N)
	#resize_image()
	solver.initialize()

#func _on_steps_per_second_slider_drag_ended(value_changed: bool) -> void:
	#steps_per_second = %StepsPerSecondSlider.value
	#%StepsPerSecondLabel.text = "Speed: %.2f" % steps_per_second

func _on_restart_button_pressed() -> void:
	solver.initialize()

func _on_solver_option_item_selected(index: int) -> void:
	solver = $Solvers.get_child(index)
	solver.initialize()
# TODO handle GUI inputs
