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

dtype_names_python: dict[DType, str] = {
    dtype: f"np.{dtype.name.lower()}"
    for dtype in DType
}
dtype_names_python[DType.Bool] = "bool"

dtype_names_nd: dict[DType, str] = {
    dtype: f"nd.{dtype.name}"
    for dtype in DType
}

@dataclass
class Test:
    name: str
    np_code: str
    nd_code: str

@dataclass
class Arg:
    def python(self):
        raise Exception

    def nd(self):
        raise Exception

@dataclass
class Grid(Arg):
    shape: tuple
    dtype: DType

    def python(self):
        return f"np.ones({self.shape}, {dtype_names_python[self.dtype]})"

    def nd(self):
        return f"nd.ones({list(self.shape)}, {dtype_names_nd[self.dtype]})"

def to_python_kwargs(kwargs):
    return "".join(f'{name}={value.python()}, ' for name, value in kwargs.items())

def to_godot_args(kwargs):
    return "".join(f'{value.nd()}, ' for name, value in kwargs.items())

def as_file_path(string):
    if os.path.isfile(string):
        return pathlib.Path(string)
    else:
        raise FileNotFoundError(string)

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--godot', type=as_file_path, required=True, help='Godot binary location')

def main():
    args = arg_parser.parse_args()

    test_code_python = \
"""import numpy as np
import time
_timer = time.perf_counter
"""

    test_code_godot = "extends Node\n\n"

    tests: list[Test] = []

    def append_test_func_python(name, args, stmt):
        nonlocal test_code_python
        test_code_python = test_code_python + \
f"""
def {name}({args}n):
    _t0 = _timer()
    for _i in range(n):
        {stmt}
    _t1 = _timer()
    return _t1 - _t0
"""

    def append_test_func_godot(name, args, stmt):
        nonlocal test_code_godot
        test_code_godot = test_code_godot + \
f"""
func __{name}({args}n):
    var _t0 = Time.get_ticks_usec()
    for _i in n:
        {stmt}
    var _t1 = Time.get_ticks_usec()
    return _t1 - _t0
"""

    def append_normal_test_func(name, function, args):
        append_test_func_python(name, args, f"np.{function}({args})")
        append_test_func_godot(name, args, f"nd.{function}({args})")

    def append_test(test_name, function_name, kwargs: dict[str, Arg], n: int):
        tests.append(Test(
            name=f"{test_name}",
            np_code=f"gen.tests.{function_name}({to_python_kwargs(kwargs)}n={n})",
            nd_code=f"print(__{function_name}({to_godot_args(kwargs)}{n}))",
        ))

    normal_n = 1_000
    all_dtypes: list[DType] = list(DType)

    for un_function_name in [
        "positive",
        "negative",
        "square",
        "sqrt",
        "abs",
        "sign",
        "sin", "cos", "tan",
        "asin", "acos", "atan",
        "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh",
    ]:
        append_normal_test_func(un_function_name, un_function_name, "x, ")
        for dtype in all_dtypes:
            for s in [1, 5, 500]:
                s2 = s * s
                append_test(f"{un_function_name}_{dtype.name}_{s2}", un_function_name, {"x": Grid((s2,), dtype)}, n=normal_n // s)

    for bin_function_name in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "pow",
    ]:
        append_normal_test_func(bin_function_name, bin_function_name, "x, y, ")
        for dtype in all_dtypes:
            for s in [1, 5, 500]:
                s2 = s * s
                append_test(f"{bin_function_name}_{dtype.name}_{s*s}", bin_function_name, {"x": Grid((s, s), dtype), "y": Grid((s, s), dtype)}, n=normal_n // s)

    py_test_file_path = pathlib.Path(__file__).parent / "gen" / "tests.py"
    py_test_file_path.parent.mkdir(exist_ok=True)
    py_test_file_path.write_text(test_code_python)

    start_numdot_tests_string = "Starting NumDot tests..."
    test_code_godot += \
f"""
func _ready():
    print("{start_numdot_tests_string}")
"""
    for test in tests:
        test_code_godot += f"    {test.nd_code}\n"

    godot_test_file_path = pathlib.Path(__file__).parent.parent / "demo" / "tests" / "run_tests.gd"
    godot_test_file_path.write_text(test_code_godot)

    print(f"Running {len(tests)} tests in godot...")
    demo_path = pathlib.Path(__file__).parent / ".." / "demo"
    gd_out = subprocess.check_output(
        [args.godot, "--path", demo_path, "--headless", "res://tests/run_tests.tscn", "--quit"],
        encoding='UTF-8'
    )
    gd_out_lines = gd_out[gd_out.index(start_numdot_tests_string):].split("\n")[1:]
    results_gd = {
        test.name: int(line) / 1000_1000
        for line, test in zip(gd_out_lines, tests)
    }

    print(f"Running {len(tests)} tests in python...")
    import gen.tests
    results = dict()
    for test in tests:
        try:
            results[test.name] = eval(test.np_code)
        except KeyboardInterrupt:
            raise
        except Exception:
            results[test.name] = -1

    df = pd.DataFrame({"numpy": results, "numdot": results_gd})
    df.to_csv("gen/results.csv")
    print("gen/results.csv")

if __name__ == "__main__":
    main()
