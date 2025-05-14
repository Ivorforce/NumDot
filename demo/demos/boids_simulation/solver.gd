extends Node
class_name BoidsSolver

@export var params: Node2D

func initialize() -> void:
	pass

func simulation_step(delta: float, velocity: Vector2, speed: float,
 visual_range: float, separation_weight: float, alignment_weight: float,
 cohesion_weight: float, position: Vector2, boid_sprite: Sprite2D, boids: Array) -> void:
	pass

func update_boids_noise(boids: Array, speed: float, delta: float) -> void:
	pass
