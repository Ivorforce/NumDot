extends Node

func _ready() -> void:
	$SieveOfEratosthenes.run_benchmark()
	$GeneratePolygon.run_benchmark()
	$Math.run_benchmark()
	$ScalarMath.run_benchmark()
	$Reductions.run_benchmark()
	$Matrix.run_benchmark()
