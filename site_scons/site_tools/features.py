import dataclasses
import enum
import itertools
import json
import os
import pathlib
import platform
import re
import sys
from typing import Collection

from SCons.Action import Action
from SCons.Builder import Builder
from SCons.Errors import UserError
from SCons.Script import ARGUMENTS
from SCons.Tool import Tool
from SCons.Variables import BoolVariable, EnumVariable, PathVariable
from SCons.Variables.BoolVariable import _text2bool

class DType(enum.Enum):
	Bool = 0
	Float32 = 1
	Float64 = 2
	Complex64 = 3
	Complex128 = 4
	Int8 = 5
	Int16 = 6
	Int32 = 7
	Int64 = 8
	UInt8 = 9
	UInt16 = 10
	UInt32 = 11
	UInt64 = 12

code_to_dtype = {
	"?": DType.Bool,
	"b": DType.Int8,
	"h": DType.Int16,
	"i": DType.Int32,
	"l": DType.Int64,
	"q": DType.Int64,
	"n": DType.Int64,
	"p": DType.Int64,
	"B": DType.UInt8,
	"H": DType.UInt16,
	"I": DType.UInt32,
	"L": DType.UInt64,
	"Q": DType.UInt64,
	"N": DType.UInt64,
	"P": DType.UInt64,
	# "e": "float16_t",
	"f": DType.Float32,
	"d": DType.Float64,
	# "g": "float128_t",
	"F": DType.Complex64,
	"D": DType.Complex128,
	# "G": "std::complex<long double_t>",
	# "S": "bytes",
	# "U": "str",
	# "V": "void",
	# "O": "object",
	# "M": "datetime64",
	# "m": "timedelta64",
}

dtype_to_c_type = {
	DType.Bool: "bool",
	DType.Int8: "int8_t",
	DType.Int16: "int16_t",
	DType.Int32: "int32_t",
	DType.Int64: "int64_t",
	DType.UInt8: "uint8_t",
	DType.UInt16: "uint16_t",
	DType.UInt32: "uint32_t",
	DType.UInt64: "uint64_t",
	DType.Float32: "float_t",
	DType.Float64: "double_t",
	DType.Complex64: "std::complex<float_t>",
	DType.Complex128: "std::complex<double_t>",
}

complex_dtypes = {
	DType.Complex64,
	DType.Complex128
}

@dataclasses.dataclass
class UFuncSpecialization:
	output: DType
	input: tuple[DType, ...]

	@staticmethod
	def parse(code: str) -> "UFuncSpecialization":
		parts = code.split("->")
		input_str = parts[0]
		output_code = code_to_dtype[parts[1]] if len(parts) > 1 else None
		return UFuncSpecialization(
			output=output_code,
			input=tuple(code_to_dtype[input_str] for input_str in input_str)
		)

def exists(env):
	return True

def options(opts):
	pass

def make_module(env, sources, module_name: str, ufuncs_json: dict):
	namespace_name = f"va::vfunc::{module_name}"

	declare_str = ""
	configure_str = ""
	for ufunc_obj in ufuncs_json["vfuncs"]:
		ufunc_name = ufunc_obj["name"]
		specializations = ufunc_obj["specializations"]
		nin = specializations[0].index("->") if "->" in specializations[0] else len(specializations[0])
		vargs = ufunc_obj["vargs"] if "vargs" in ufunc_obj else []
		vargs_part = "".join(f", {varg}" for varg in vargs)

		declare_str += f"\tDECLARE_VFUNC({ufunc_name});\n"

		covered_types: set[tuple[DType, ...]] = set()
		for specialization_str in specializations:
			specialization = UFuncSpecialization.parse(specialization_str)

			if specialization.input in covered_types:
				# Some types exist twice for some reason, see above.
				continue
			covered_types.add(specialization.input)

			# FIXME Need to test how these functions are computed with complex dtypes in numpy.
			if any(dtype in complex_dtypes for dtype in specialization.input) and (
				ufunc_name == "minimum"
				or ufunc_name == "maximum"
				or ufunc_name == "max"
				or ufunc_name == "min"
				or ufunc_name == "mean"
				or ufunc_name == "median"
				or ufunc_name == "variance"
				or ufunc_name == "standard_deviation"
				or ufunc_name == "rint"
				or ufunc_name == "less_equal"
				or ufunc_name == "less"
				or ufunc_name == "greater"
				or ufunc_name == "greater_equal"
			):
				continue

			input_types_cpp = "".join(f", {dtype_to_c_type[dtype]}" for dtype in specialization.input)
			output_type_cpp = f", {dtype_to_c_type[specialization.output]}" if specialization.output is not None else ""

			configure_str += f"\tadd_native<{ufunc_name}{output_type_cpp}{input_types_cpp}{vargs_part}>(tables::{ufunc_name});\n"

		for cast_str in ufunc_obj["casts"]:
			in_str, model_str = cast_str.split("->", maxsplit=1)
			model = UFuncSpecialization.parse(model_str)
			in_types = tuple(code_to_dtype[dtype_str] for dtype_str in in_str)

			assert model.input in covered_types, f"{ufunc_name} has a cast {in_str} to {model_str} even though the specialization does not exist"
			assert in_types not in covered_types, f"{ufunc_name} has a cast {in_str} to {model_str} even though it was already specialized for these types"
			assert len(model.input) == nin
			assert len(in_types) == nin

			template_args_cpp = ", ".join(
				dtype_to_c_type[dtype]
				for dtype in itertools.chain(in_types, model.input)
			)
			configure_str += f"\tadd_cast<{template_args_cpp}>(tables::{ufunc_name});\n"

	ifndef_macro = f"VATENSOR_UFUNC_{module_name}_HPP".upper()
	hpp_contents = \
f"""
#ifndef {ifndef_macro}
#define {ifndef_macro}

namespace {namespace_name} {{
	void configure();
}}

#endif // {ifndef_macro}
""".strip()

	cpp_contents = \
f"""
#include "{module_name}.hpp"

#include "vatensor/varray.hpp"
#include "vatensor/vfunc/vfuncs.hpp"
#include "vatensor/vfunc/tables.hpp"
#include "vatensor/vfunc/arch_util.hpp"
#include "xtensor/xoperation.hpp"

namespace {namespace_name} {{
{declare_str}}}

void {namespace_name}::configure() {{
{configure_str}}}
""".strip()

	gen_path = pathlib.Path(f"src/vatensor/gen/")
	gen_path.mkdir(exist_ok=True)

	pathlib.Path(gen_path / f"{module_name}.hpp").write_text(hpp_contents)
	cpp_path = pathlib.Path(gen_path / f"{module_name}.cpp")
	cpp_path.write_text(cpp_contents)

	# TODO Shouldn't use glob for this lol
	sources.append(env.Glob(str(cpp_path)))

def generate(env, sources):
	with pathlib.Path("configure/vfuncs.json").open("r") as f:
		ufuncs_json = json.load(f)

	make_module(env, sources, module_name="base", ufuncs_json=ufuncs_json)
