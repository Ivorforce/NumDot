import enum
import pathlib
import importlib
from dataclasses import dataclass

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
	np.dtype(np.complex64): "Complex64",
	np.dtype(np.complex128): "Complex128",
}

@dataclass
class Test:
	name: str
	np_code: str
	nd_code: str
	gd_code: str

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

def func_to_gdscript(func, nin, is_complex):
	if nin == 1:
		if func == "positive":
			return "{} = +{}"
		if func == "negative":
			return "{} = -{}"
		if func == "square":
			return "{0} = {1} * {1}"
		if func == "bitwise_not":
			return "{} = ~{}"
		if func == "logical_not":
			return "{} = not {}"
		if func == "rad2deg":
			return "{} = rad_to_deg({})"
		if func == "deg2rad":
			return "{} = deg_to_rad({})"
		if func == "rint":
			return "{} = round({})"
		if func == "trunc":
			return "{} = int({})"
		if is_complex and (
				"sin" in func
				or "cos" in func
				or "tan" in func
				or func == "sqrt"
		):
			return f"{{0}} = Vector2({func}({{1}}.x), {func}({{1}}.y))"
		if is_complex and func == "abs":
			return f"{{0}} = Vector2(sqrt({{1}}.x * {{1}}.x + {{1}}.y * {{1}}.y), 0)"  # TODO should be set in a float array to be fair
		return f"{{}} = {func}({{}})"
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
			return f"{{0}} = Vector2({func}({{1}}.x, {{2}}.x), {func}({{1}}.y, {{2}}.y))"
		if func == "add":
			return "{} = {} + {}"
		if func == "subtract":
			return "{} = {} - {}"
		if func == "multiply":
			return "{} = {} * {}"
		if func == "divide":
			return "{} = {} / {}"
		if func == "greater":
			return "{} = {} > {}"
		if func == "greater_equal":
			return "{} = {} <= {}"
		if func == "less":
			return "{} = {} < {}"
		if func == "less_equal":
			return "{} = {} >= {}"
		if func == "equal":
			return "{} = {} == {}"
		if func == "not_equal":
			return "{} = {} != {}"
		if func == "logical_and":
			return "{} = {} and {}"
		if func == "logical_or":
			return "{} = {} or {}"
		if func == "logical_xor":
			return "{} = bool({}) != bool({})"
		if func == "bitwise_and":
			return "{} = {} & {}"
		if func == "bitwise_or":
			return "{} = {} | {}"
		if func == "bitwise_xor":
			return "{} = {} ^ {}"
		if func == "bitwise_left_shift":
			return "{} = {} << {}"
		if func == "bitwise_right_shift":
			return "{} = {} >> {}"
		if func == "matmul":
			return None
		if func == "remainder":
			return None  # TODO different for float vs int
		return f"{{}} = {func}({{}}, {{}})"
	else:
		raise NotImplementedError


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

def make_test_func_gd(name, args, stmt):
	return \
f"""
func __{name}({args}n):
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
	if any(arg.dtype in [np.dtype(np.complex64), np.dtype(np.complex128)] for arg in kwargs.values()):
		function_name = function_name + "_complex"

	args_str = "".join(f'to_packed({nd_arg_to_str(value)}), ' for name, value in kwargs.items())
	return f"\tprint(\"{test_number} \", __{function_name}({args_str}{n}))"

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
	test_code_gd = \
"""extends Node\n\n

func to_packed(array: NDArray):
	var dtype := array.dtype()
	if dtype == nd.Int8:
		return array.to_packed_int32_array()
	if dtype == nd.Int16:
		return array.to_packed_int32_array()
	if dtype == nd.Int32:
		return array.to_packed_int64_array()
	if dtype == nd.Int64:
		return array.to_packed_int64_array()
	if dtype == nd.UInt8:
		return array.to_packed_byte_array()
	if dtype == nd.UInt16:
		return array.to_packed_int32_array()
	if dtype == nd.UInt32:
		return array.to_packed_int32_array()
	if dtype == nd.UInt64:
		return array.to_packed_int64_array()
	if dtype == nd.Float32:
		return array.to_packed_float32_array()
	if dtype == nd.Float64:
		return array.to_packed_float64_array()
	if dtype == nd.Bool:
		return array.to_packed_byte_array()
	if dtype == nd.Complex64:
		return nd.complex_as_vector(array).to_packed_vector2_array()
	if dtype == nd.Complex128:  # Not technically correct because it's not double, but eh...
		return nd.complex_as_vector(array).to_packed_vector2_array()
	assert(false)
"""

	tests: list[Test] = []

	def append_normal_test_func(name, function, args):
		nonlocal test_code_np
		args_str = "".join(f"{arg}, " for arg in args)
		test_code_np += make_test_func_np(name, args_str, f"np.{function}({args_str})")

		nonlocal test_code_nd
		test_code_nd += make_test_func_gd(name, args_str, f"\t\tnd.{function}({args_str})")

		nonlocal test_code_gd
		for is_complex in [True, False]:
			gdscript_code = func_to_gdscript(function, len(args), is_complex)
			if gdscript_code is None:
				continue

			test_code = "\t\tfor i in x.size():\n\t\t\t"""
			test_code += gdscript_code.format(*[f"{arg}[i]" for arg in [args[0], *args]])

			args_str_def = "".join(f"{arg}, " for arg in args)
			test_code_gd += make_test_func_gd(name + ("_complex" if is_complex else ""), args_str_def, test_code)

	current_test_number = 0

	def append_test(test_name, function_name, kwargs: dict[str, Arg], n: int):
		nonlocal current_test_number

		tests.append(Test(
			name=f"{test_name}",
			np_code=make_np_call(function_name, kwargs, n),
			# TODO Should find a better way to cascade-remove tests we aren't compatible with
			nd_code=make_nd_call(function_name, current_test_number, kwargs, n) if f"__{function_name}(" in test_code_nd else None,
			gd_code=make_gd_call(function_name, current_test_number, kwargs, n) if f"__{function_name}(" in test_code_gd else None,
		))
		current_test_number += 1

	normal_n = 40_000

	# TODO No support for reductions yet
	# TODO Should automatically (?) determine what NumDot has?
	for un_function_name in [
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
	]:
		np_ufunc = eval(f"np.{un_function_name}")
		print(un_function_name)
		ufunc_args = "xyz"[:np_ufunc.nin]

		append_normal_test_func(un_function_name, un_function_name, ufunc_args)

		for type_str in np_ufunc.types:
			dtype_in = np.dtype(type_str[0])
			dtype_out = np.dtype(type_str[-1])
			if dtype_in != dtype_out:
				continue  # TODO Our godot test generation can't handle this yet lol
			if dtype_in not in dtype_names_nd:
				continue  # Skip what weNumDot doesn't support anyway.

			for s in [50, 1_000, 20000]:
				append_test(
					f"{un_function_name}_{dtype_in}_{s}",
					un_function_name,
					{arg: Full(s, value="0.5", dtype=dtype_in) for arg in ufunc_args},
					n=normal_n // s
				)

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
