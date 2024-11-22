import enum
import pathlib
import importlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os

import argparse
import subprocess

class DType(enum.Enum):
	Int8 = 0
	Int16 = 1
	Int32 = 2
	Int64 = 3
	UInt8 = 4
	UInt16 = 5
	UInt32 = 6
	UInt64 = 7
	Float32 = 8
	Float64 = 9
	Bool = 10
	Complex64 = 11
	Complex128 = 12

dtype_names_np: dict[DType, str] = {
	dtype: f"np.{dtype.name.lower()}"
	for dtype in DType
}
dtype_names_np[DType.Bool] = "bool"

dtype_names_nd: dict[DType, str] = {
	dtype: f"nd.{dtype.name}"
	for dtype in DType
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
	dtype: DType

def as_file_path(string):
	if os.path.isfile(string):
		return pathlib.Path(string)
	else:
		raise FileNotFoundError(string)

def unary_func_to_gdscript(func, is_complex):
	if func == "positive":
		return "{} = +{}"
	if func == "negative":
		return "{} = -{}"
	if func == "square":
		return "{0} = {1} * {1}"
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

def binary_func_to_gdscript(func, is_complex):
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
	if func == "equal":
		return "{} = {} == {}"
	return f"{{}} = {func}({{}}, {{}})"


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
			return f"np.full([{arg.size}], fill_value={arg.value}, dtype={dtype_names_np[arg.dtype]})"
		raise Exception()

	args_str = "".join(f'{name}={arg_to_str(value)}, ' for name, value in kwargs.items())
	return f"gen.tests.{function_name}({args_str}n={n})"

def nd_arg_to_str(arg: Arg):
	if isinstance(arg, Full):
		return f"nd.full([{arg.size}], {arg.value}, {dtype_names_nd[arg.dtype]})"
	raise Exception()

def make_nd_call(function_name, kwargs: dict[str, Arg], n: int):
	args_str = "".join(f'{nd_arg_to_str(value)}, ' for name, value in kwargs.items())
	return f"\tprint(__{function_name}({args_str}{n}))"

def make_gd_call(function_name, kwargs: dict[str, Arg], n: int):
	if any(arg.dtype in [DType.Complex64, DType.Complex128] for arg in kwargs.values()):
		function_name = function_name + "_complex"

	args_str = "".join(f'to_packed({nd_arg_to_str(value)}), ' for name, value in kwargs.items())
	return f"\tprint(__{function_name}({args_str}{n}))"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--godot', type=as_file_path, required=True, help='Godot binary location')

def main():
	args = arg_parser.parse_args()

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

	def make_normal_test_func_gd(name, function, replacer, args):
		nonlocal test_code_gd
		for is_complex in [True, False]:
			test_code = "\t\tfor i in x.size():\n\t\t\t"""

			test_code += replacer(function, is_complex).format(*[f"{arg}[i]" for arg in [args[0], *args]])

			args_str_def = "".join(f"{arg}, " for arg in args)
			test_code_gd += make_test_func_gd(name + ("_complex" if is_complex else ""), args_str_def, test_code)

	def append_test(test_name, function_name, kwargs: dict[str, Arg], n: int):
		tests.append(Test(
			name=f"{test_name}",
			np_code=make_np_call(function_name, kwargs, n),
			nd_code=make_nd_call(function_name, kwargs, n),
			gd_code=make_gd_call(function_name, kwargs, n)
		))

	normal_n = 40_000
	all_dtypes: list[DType] = list(DType)

	for un_function_name in [
		# "positive",
		# "negative",
		"square",
		"sqrt",
		"abs",
		# "sign",
		"sin", "cos", "tan",
		"asin", "acos", "atan",
		"sinh", "cosh", "tanh",
		"asinh", "acosh", "atanh",
	]:
		append_normal_test_func(un_function_name, un_function_name, "x")
		make_normal_test_func_gd(un_function_name, un_function_name, unary_func_to_gdscript, "x")

		for dtype in all_dtypes:
			for s in [50, 1_000, 20000]:
				append_test(f"{un_function_name}_{dtype.name}_{s}", un_function_name, {"x": Full(s, value="0.5", dtype=dtype)}, n=normal_n // s)

	for bin_function_name in [
		"add",
		"subtract",
		"multiply",
		"divide",
		"pow",
		# "greater",
		# "equal",
		# "minimum",
		# "maximum",
	]:
		append_normal_test_func(bin_function_name, bin_function_name, "xy")
		make_normal_test_func_gd(bin_function_name, bin_function_name, binary_func_to_gdscript, "xy")

		for dtype in all_dtypes:
			for s in [50, 1_000, 20000]:
				append_test(f"{bin_function_name}_{dtype.name}_{s}", bin_function_name, {"x": Full(s, value="0.5", dtype=dtype), "y": Full(s, value="2", dtype=dtype)}, n=normal_n // s)

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
			text += f"{getattr(test, test_prop)}\n"

		godot_test_file_path = pathlib.Path(__file__).parent.parent / "demo" / "tests" / "run_tests.gd"
		godot_test_file_path.write_text(text)

		print(f"Running {len(tests)} tests in godot...")

		demo_path = pathlib.Path(__file__).parent / ".." / "demo"
		gd_out = subprocess.check_output(
			[args.godot, "--path", demo_path, "--headless", "res://tests/run_tests.tscn", "--quit"],
			encoding='UTF-8'
		)
		gd_out_lines = gd_out[gd_out.index(start_numdot_tests_string):].split("\n")[1:]

		def parse_seconds(s):
			try:
				return int(s) / 1_000_000
			except:
				return np.nan

		return {
			# microseconds to seconds
			test.name: parse_seconds(line)
			for line, test in zip(gd_out_lines, tests)
		}

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
