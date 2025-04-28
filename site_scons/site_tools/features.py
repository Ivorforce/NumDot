import dataclasses
import enum
import itertools
import json
import pathlib
import re

from SCons.Action import Action
from SCons.Builder import Builder
from SCons.Errors import UserError
from SCons.Script import ARGUMENTS
from SCons.Tool import Tool
from SCons.Variables import BoolVariable, EnumVariable, PathVariable
from SCons.Variables.BoolVariable import _text2bool
from typing import Optional


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

	def __str__(self):
		return self.name

	def __repr__(self):
		return f"DType.{self.name}"

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
class VFuncInfo:
	name: str
	vargs: list[str]

@dataclasses.dataclass
class VFuncSpecialization:
	vfunc_name: str

	output: DType
	input: tuple[DType, ...]
	model_input: Optional[tuple[DType, ...]]

	is_enabled: bool = True

@dataclasses.dataclass
class Features:
	all: list[VFuncSpecialization]
	vfunc_infos: dict[str, VFuncInfo]


	@staticmethod
	def from_json_obj(object: dict) -> "Features":
		features = Features(all=[], vfunc_infos=dict())

		for vfunc_obj in object["vfuncs"]:
			for code in vfunc_obj["specializations"]:
				input_str, *rest_parts = code.split("->")
				features.vfunc_infos.setdefault(vfunc_obj["name"], VFuncInfo(
					name=vfunc_obj["name"],
					vargs=vfunc_obj["vargs"] if "vargs" in vfunc_obj else []
				))
				features.all.append(
					VFuncSpecialization(
						vfunc_name=vfunc_obj["name"],
						output=code_to_dtype[rest_parts[0]] if len(rest_parts) > 0 else None,
						input=tuple(code_to_dtype[input_str] for input_str in input_str),
						model_input=None
					)
				)

			for code in vfunc_obj["casts"]:
				input_str, model_input_str, *rest_parts = code.split("->")
				features.vfunc_infos.setdefault(vfunc_obj["name"], VFuncInfo(
					name=vfunc_obj["name"],
					vargs=vfunc_obj["vargs"] if "vargs" in vfunc_obj else []
				))
				features.all.append(
					VFuncSpecialization(
						vfunc_name=vfunc_obj["name"],
						output=code_to_dtype[rest_parts[0]] if len(rest_parts) > 0 else None,
						input=tuple(code_to_dtype[input_str] for input_str in input_str),
						model_input=tuple(code_to_dtype[input_str] for input_str in model_input_str)
					)
				)

		return features

	def name_matching(self, fn) -> list[VFuncSpecialization]:
		return [f for f in self.all if fn(f.vfunc_name)]

	@property
	def bitwise(self) -> list[VFuncSpecialization]:
		return self.name_matching(lambda name: name.startswith("bitwise"))

	@property
	def logical(self) -> list[VFuncSpecialization]:
		return self.name_matching(lambda name: name.startswith("logical"))

	@property
	def random(self) -> list[VFuncSpecialization]:
		return self.name_matching(lambda name: name.startswith("random"))

	@property
	def trigonometry(self) -> list[VFuncSpecialization]:
		return self.name_matching(lambda name: re.match(r"a?(sin|cos|tan)h?", name))

	@property
	def enabled(self) -> list[VFuncSpecialization]:
		return [f for f in self.all if f.is_enabled]

	@property
	def disabled(self) -> list[VFuncSpecialization]:
		return [f for f in self.all if not f.is_enabled]

	def find(self, feature=None):
		if isinstance(feature, list) or isinstance(feature, tuple):
			return [
				single_feature
				for feature_part in feature
				for single_feature in self.find(feature_part)
			]
		if isinstance(feature, str):
			return [x for x in self.all if x.vfunc_name == feature]
		elif isinstance(feature, VFuncInfo):
			return [x for x in self.all if x == feature]
		elif isinstance(feature, VFuncSpecialization):
			return [feature]
		else:
			raise ValueError

	def disable(self, *args, **kwargs):
		for specialization in self.find(*args, **kwargs):
			specialization.is_enabled = False

	def enable(self, *args, **kwargs):
		for specialization in self.find(*args, **kwargs):
			specialization.is_enabled = True

def exists(env):
	return True

def options(opts):
	opts.Add(
		PathVariable(
			key="numdot_config",
			help="Path to a .py file that sets up custom NumDot configuration.",
			default=None,
		)
	)

def make_module(env, sources, module_name: str, features: Features):
	namespace_name = f"va::vfunc::{module_name}"

	declare_str = ""
	for vfunc in features.vfunc_infos.values():
		declare_str += f"\tDECLARE_VFUNC({vfunc.name});\n"

	configure_str = ""
	for specialization in features.all:
		if not specialization.is_enabled:
			continue


		covered_types: set[tuple[DType, ...]] = set()
		if specialization.input in covered_types:
			# Some types exist twice for some reason, see above.
			continue
		covered_types.add(specialization.input)

		vfunc = features.vfunc_infos[specialization.vfunc_name]

		if specialization.model_input is None:
			# FIXME Need to test how these functions are computed with complex dtypes in numpy.
			if any(dtype in complex_dtypes for dtype in specialization.input) and (
				vfunc.name == "minimum"
				or vfunc.name == "maximum"
				or vfunc.name == "max"
				or vfunc.name == "min"
				or vfunc.name == "mean"
				or vfunc.name == "median"
				or vfunc.name == "variance"
				or vfunc.name == "standard_deviation"
				or vfunc.name == "rint"
				or vfunc.name == "less_equal"
				or vfunc.name == "less"
				or vfunc.name == "greater"
				or vfunc.name == "greater_equal"
				or vfunc.name == "round"
			):
				continue

			input_types_cpp = "".join(f", {dtype_to_c_type[dtype]}" for dtype in specialization.input)
			output_type_cpp = f", {dtype_to_c_type[specialization.output]}" if specialization.output is not None else ""

			vargs_part = "".join(f", {varg}" for varg in vfunc.vargs)
			configure_str += f"\tadd_native<{vfunc.name}{output_type_cpp}{input_types_cpp}{vargs_part}>(tables::{vfunc.name});\n"

		else:
			template_args_cpp = ", ".join(
				dtype_to_c_type[dtype]
				for dtype in itertools.chain(specialization.input, specialization.model_input)
			)
			configure_str += f"\tadd_cast<{template_args_cpp}>(tables::{vfunc.name});\n"

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
#include "xtensor/core/xoperation.hpp"

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
		features = Features.from_json_obj(json.load(f))

	if 'numdot_config' in env and env['numdot_config']:
		custom_config_path = pathlib.Path(env['numdot_config'])
		assert custom_config_path.suffix == ".py"
		# -3 to cut the '.py'
		custom_config_tool = Tool(custom_config_path.name[:-3], toolpath=[custom_config_path.parent])

		custom_config_tool.generate(env, features)

	make_module(env, sources, module_name="base", features=features)
