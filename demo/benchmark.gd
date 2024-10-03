class_name Benchmark
extends Node

var current_section_name: String
var current_section_start_time_us: int

func begin_section(name: String):
	current_section_name = name
	current_section_start_time_us = Time.get_ticks_usec()

func store_result():
	var time_diff := Time.get_ticks_usec() - current_section_start_time_us
	print("%s: %s" % [current_section_name, str(time_diff)])

func end():
	print()
