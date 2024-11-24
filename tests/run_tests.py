import enum
import pathlib
import importlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import os

import argparse
import subprocess

dtype_names_nd: dict[np.dtype, str] = {
	np.dtype(np.int8): "Int8",
	np.dtype(np.int16): "Int16",
	np.dtype(np.int32): "Int32",
	np.dtype(np.int64): "Int64",
	np.dtype(np.uint8): "UInt8",
	np.dtype(np.uint16): "UInt16",
	np.dtype(np.uint32): "UInt32",
	np.dtype(np.uint64): "UInt64",
	np.dtype(np.bool): "Bool",
	np.dtype(np.float32): "Float32",
	np.dtype(np.float64): "Float64",
	np.dtype(np.complex64): "Complex64",
	np.dtype(np.complex128): "Complex128",
}

@dataclass
class Test:
	name: str
	np_code: Optional[str] = None
	nd_code: Optional[str] = None
	gd_code: Optional[str] = None

@dataclass
class Arg:
	pass

@dataclass
class Full(Arg):
	size: int
	value: str
	dtype: np.dtype

def as_file_path(string):
	if os.path.isfile(string):
		return pathlib.Path(string)
	else:
		raise FileNotFoundError(string)

def _func_to_gdscript(func, nin, dtype_out):
	# TODO This is a bad proxy for is_complex...
	is_complex = dtype_out == np.dtype(np.complex64) or dtype_out == np.dtype(np.complex128)

	if nin == 1:
		if func == "positive":
			return "+{}"
		if func == "negative":
			return "-{}"
		if func == "square":
			return "{1} * {1}"
		if func == "bitwise_not":
			return "~{}"
		if func == "logical_not":
			return "not {}"
		if func == "rad2deg":
			return "rad_to_deg({})"
		if func == "deg2rad":
			return "deg_to_rad({})"
		if func == "rint":
			return "round({})"
		if func == "trunc":
			return "int({})"
		if is_complex and (
				"sin" in func
				or "cos" in func
				or "tan" in func
				or func == "sqrt"
				or func == "exp"
				or func == "log"
		):
			return f"Vector2({func}({{1}}.x), {func}({{1}}.y))"
		if is_complex and func == "abs":
			return f"Vector2(sqrt({{1}}.x * {{1}}.x + {{1}}.y * {{1}}.y), 0)"  # TODO should be set in a float array to be fair
		return f"{func}({{}})"
	elif nin == 2:
		if func == "maximum":
			func = "max"
		if func == "minimum":
			func = "min"
		if is_complex and (
				func == "pow"
				or func == "max"
				or func == "min"
		):
			return f"Vector2({func}({{1}}.x, {{2}}.x), {func}({{1}}.y, {{2}}.y))"
		if func == "add":
			return "{} + {}"
		if func == "subtract":
			return "{} - {}"
		if func == "multiply":
			return "{} * {}"
		if func == "divide":
			return "{} / {}"
		if func == "greater":
			return "{} > {}"
		if func == "greater_equal":
			return "{} <= {}"
		if func == "less":
			return "{} < {}"
		if func == "less_equal":
			return "{} >= {}"
		if func == "equal":
			return "{} == {}"
		if func == "not_equal":
			return "{} != {}"
		if func == "logical_and":
			return "{} and {}"
		if func == "logical_or":
			return "{} or {}"
		if func == "logical_xor":
			return "bool({}) != bool({})"
		if func == "bitwise_and":
			return "{} & {}"
		if func == "bitwise_or":
			return "{} | {}"
		if func == "bitwise_xor":
			return "{} ^ {}"
		if func == "bitwise_left_shift":
			return "{} << {}"
		if func == "bitwise_right_shift":
			return "{} >> {}"
		if func == "matmul":
			return None
		if func == "remainder":
			if dtype_out in [np.dtype(np.float32), np.dtype(np.float64)]:
				return "fmod({}, {})"
			else:
				return "{} % {}"
		return f"{func}({{}}, {{}})"
	else:
		raise NotImplementedError

def func_to_gdscript(func, nin, dtype_out):
	create_string = _func_to_gdscript(func, nin, dtype_out)
	if create_string is None:
		return None

	if dtype_out == np.dtype(np.bool):
		# Bools don't have dedicated packed arrays, so we're using byte array
		create_string = f"int({create_string})"
	if "{}" in create_string:
		return "{} = " + create_string
	else:
		return "{0} = " + create_string

def make_test_func_np(name, args, stmt):
	return \
f"""
def {name}({args}n):
\t_t0 = _timer()
\tfor _n in range(n):
\t\t{stmt}
\t_t1 = _timer()
\treturn _t1 - _t0
"""

def make_test_func_nd(name, args, stmt):
	return \
f"""
func __{name}({args}n):
\tvar _t0 = Time.get_ticks_usec()
\tfor _n in n:
{stmt}
\tvar _t1 = Time.get_ticks_usec()
\treturn _t1 - _t0
"""

def make_test_func_gd(name, args, dtype_out, stmt):
	# We don't count array creation to the time because the user may assign to the array itself, or pre-allocate.
	return \
f"""
func __{name}({args}n):
\tvar out = TestUtil.to_packed(nd.full([x.size()], 0, nd.{dtype_names_nd[dtype_out]}))
\tout.resize(x.size())
\tvar _t0 = Time.get_ticks_usec()
\tfor _n in n:
{stmt}
\tvar _t1 = Time.get_ticks_usec()
\treturn _t1 - _t0
"""

def make_np_call(function_name, kwargs: dict[str, Arg], n: int):
	def arg_to_str(arg: Arg):
		if isinstance(arg, Full):
			return f"np.full([{arg.size}], fill_value={arg.value}, dtype=np.{arg.dtype})"
		raise Exception()

	args_str = "".join(f'{name}={arg_to_str(value)}, ' for name, value in kwargs.items())
	return f"gen.tests.{function_name}({args_str}n={n})"

def nd_arg_to_str(arg: Arg):
	if isinstance(arg, Full):
		return f"nd.full([{arg.size}], {arg.value}, nd.{dtype_names_nd[arg.dtype]})"
	raise Exception()

def make_nd_call(function_name, test_number: int, kwargs: dict[str, Arg], n: int):
	args_str = "".join(f'{nd_arg_to_str(value)}, ' for name, value in kwargs.items())
	return f"\tprint(\"{test_number} \", __{function_name}({args_str}{n}))"

def make_gd_call(function_name, test_number: int, kwargs: dict[str, Arg], n: int):
	args_str = "".join(f'TestUtil.to_packed({nd_arg_to_str(value)}), ' for name, value in kwargs.items())
	return f"\tprint(\"{test_number} \", __{function_name}({args_str}{n}))"

TEST_UFUNCS = [
	"abs",
	"acos",
	"acosh",
	"add",
	# "all",
	# "angle",  # Not a ufunc apparently
	# "any",
	"asin",
	"asinh",
	"atan",
	"atan2",
	"atanh",
	"bitwise_and",
	"bitwise_left_shift",
	"bitwise_not",
	"bitwise_or",
	"bitwise_right_shift",
	"bitwise_xor",
	"ceil",
	# "clip",  # Not a ufunc apparently
	"cos",
	"cosh",
	# "count_nonzero",
	"deg2rad",
	"divide",
	# "dot",  # Not a ufunc apparently
	"equal",
	"exp",
	# "fft",  # Not a ufunc apparently
	"floor",
	"greater",
	"greater_equal",
	# "is_close",  # TODO Renamed
	# "is_finite",  # TODO Renamed
	# "is_inf",  # TODO Renamed
	# "is_nan",  # TODO Renamed
	"less",
	"less_equal",
	"log",
	"logical_and",
	"logical_not",
	"logical_or",
	"logical_xor",
	"matmul",
	# "max",
	"maximum",
	# "mean",  # TODO Not a ufunc
	# "median",  # TODO Not a ufunc
	# "min",  # TODO Not a ufunc
	"minimum",
	"multiply",
	"negative",
	# "norm",  # TODO Not a ufunc
	"not_equal",
	"positive",
	"pow",
	# "prod",  # TODO Not a ufunc
	"rad2deg",
	"remainder",
	"rint",
	# "round",  # TODO Not a ufunc
	"sign",
	"sin",
	"sinh",
	"sqrt",
	"square",
	# "std",  # TODO Not a ufunc
	"subtract",
	# "sum",  # TODO Not a ufunc
	"tan",
	"tanh",
	"trunc",
	# "var",  # TODO Not a ufunc
]

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--godot', type=as_file_path, required=True, help='Godot binary location')

def main():
	cli_args = arg_parser.parse_args()

	test_code_np = \
"""import numpy as np
import time
_timer = time.perf_counter
"""

	test_code_nd = "extends Node\n\n"
	test_code_gd = "extends Node\n\n"

	tests: list[Test] = []

	current_test_number = 0

	normal_n = 40_000
	added_functions = set()

	# TODO No support for reductions yet
	# TODO Should automatically (?) determine what NumDot has?
	for ufunc_name in TEST_UFUNCS:
		np_ufunc = eval(f"np.{ufunc_name}")
		ufunc_args = "xyz"[:np_ufunc.nin]

		test_function_name_untyped = f"{ufunc_name}"

		args_str = "".join(f"{arg}, " for arg in ufunc_args)
		test_code_np += make_test_func_np(test_function_name_untyped, args_str, f"np.{ufunc_name}({args_str})")
		test_code_nd += make_test_func_nd(test_function_name_untyped, args_str, f"\t\tnd.{ufunc_name}({args_str})")

		for type_str in np_ufunc.types:
			dtype_in = np.dtype(type_str[0])
			dtype_out = np.dtype(type_str[-1])

			if dtype_in not in dtype_names_nd or dtype_out not in dtype_names_nd:
				continue  # Skip what NumDot doesn't support anyway.

			test_function_name_typed = f"{ufunc_name}_{dtype_in}"

			if test_function_name_typed in added_functions:
				continue  # TODO Figure out why
			added_functions.add(test_function_name_typed)

			has_gd_test = False
			gdscript_code = func_to_gdscript(ufunc_name, len(ufunc_args), dtype_out)
			if gdscript_code is not None:
				test_code = "\t\tfor i in x.size():\n\t\t\t"""
				test_code += gdscript_code.format("out[i]", *[f"{arg}[i]" for arg in ufunc_args])

				args_str_def = "".join(f"{arg}, " for arg in ufunc_args)
				test_code_gd += make_test_func_gd(test_function_name_typed, args_str_def, dtype_out, test_code)
				has_gd_test = True

			for s in [50, 1_000, 20000]:
				test = Test(f"{ufunc_name}_{dtype_in}_{s}")
				test_kwargs = {arg: Full(s, value="1", dtype=dtype_in) for arg in ufunc_args}
				test_n = normal_n // s

				test.np_code = make_np_call(test_function_name_untyped, test_kwargs, test_n)
				test.nd_code = make_nd_call(test_function_name_untyped, current_test_number, test_kwargs, test_n)
				if has_gd_test:
					test.gd_code = make_gd_call(test_function_name_typed, current_test_number, test_kwargs, test_n)

				tests.append(test)
				current_test_number += 1


	py_test_file_path = pathlib.Path(__file__).parent / "gen" / "tests.py"
	py_test_file_path.parent.mkdir(exist_ok=True)
	py_test_file_path.write_text(test_code_np)

	def run_godot_tests(text, test_prop):
		start_numdot_tests_string = "Starting NumDot tests..."
		text += \
f"""
func _ready():
\tprint("{start_numdot_tests_string}")
"""
		for test in tests:
			if test_code := getattr(test, test_prop):
				text += f"{test_code}\n"

		godot_test_file_path = pathlib.Path(__file__).parent.parent / "demo" / "tests" / "run_tests.gd"
		godot_test_file_path.write_text(text)

		print(f"Running {len(tests)} tests in godot...")

		demo_path = pathlib.Path(__file__).parent / ".." / "demo"
		gd_out = subprocess.check_output(
			[cli_args.godot, "--path", demo_path, "--headless", "res://tests/run_tests.tscn", "--quit"],
			encoding='UTF-8'
		)
		gd_out_lines = gd_out[gd_out.index(start_numdot_tests_string):].split("\n")[1:]

		res = dict()
		for line in gd_out_lines:
			split = line.split(" ")
			if len(split) == 1:
				continue
			test_num = int(split[0])

			try:
				test_duration_s =  int(split[1]) / 1_000_000
			except:
				continue

			res[tests[test_num].name] = test_duration_s
		return res

	results_gd = run_godot_tests(test_code_gd, "gd_code")
	results_nd = run_godot_tests(test_code_nd, "nd_code")

	print(f"Running {len(tests)} tests in python...")
	import gen.tests
	results = dict()
	for test in tests:
		try:
			results[test.name] = eval(test.np_code)
		except KeyboardInterrupt:
			raise
		except Exception:
			results[test.name] = np.nan

	df = pd.DataFrame({"numpy": results, "numdot": results_nd, "godot": results_gd})
	df.to_csv("gen/results.csv")
	print("gen/results.csv")

if __name__ == "__main__":
	main()
