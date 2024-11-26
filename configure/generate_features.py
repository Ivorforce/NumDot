import dataclasses
import json
import pathlib

import numpy as np

supported_dtypes: set[np.dtype] = {
	np.dtype(np.bool),
	np.dtype(np.float32),
	np.dtype(np.float64),
	np.dtype(np.complex64),
	np.dtype(np.complex128),
	np.dtype(np.int8),
	np.dtype(np.int16),
	np.dtype(np.int32),
	np.dtype(np.int64),
	np.dtype(np.uint8),
	np.dtype(np.uint16),
	np.dtype(np.uint32),
	np.dtype(np.uint64),
}

all_features = [
	"negative",
	"sign",
	"abs",
	"square",
	"sqrt",
	"exp",
	"log",
	"rad2deg",
	"deg2rad",

	"add",
	"subtract",
	"multiply",
	"divide",
	"remainder",
	"pow",
	"minimum",
	"maximum",

	"sin",
	"cos",
	"tan",
	"asin",
	"acos",
	"atan",
	"atan2",
	"sinh",
	"cosh",
	"tanh",
	"asinh",
	"acosh",
	"atanh",

	"ceil",
	"floor",
	"trunc",
	"rint",

	"logical_not",
	"logical_and",
	"logical_or",
	"logical_xor",

	"bitwise_not",
	"bitwise_and",
	"bitwise_or",
	"bitwise_xor",
	"bitwise_left_shift",
	"bitwise_right_shift",

	"equal",
	"not_equal",
	"less",
	"less_equal",
	"greater",
	"greater_equal",
	"isnan",
	"isfinite",
	"isinf",
]

common_dtypes = dict()

def dump_dtypes(args):
	return ''.join(dtype.char for dtype in args)

@dataclasses.dataclass
class UFuncSpecialization:
	output: np.dtype
	input: tuple[np.dtype, ...]

	@staticmethod
	def parse(code: str) -> "UFuncSpecialization":
		input_str, output_str = code.split("->")
		return UFuncSpecialization(
			output=np.dtype(output_str),
			input=tuple(np.dtype(input_str) for input_str in input_str)
		)

	def dumps(self):
		return f"{dump_dtypes(self.input)}->{self.output.char}"

def main():
	all_typecodes = np.typecodes["All"]
	all_dtypes = [np.dtype(code) for code in all_typecodes]

	for dtype in all_dtypes:
		# print(f"\"{all_dtypes.char}\": \"{dtype.name}\",")
		for dtype_r in all_dtypes:
			try:
				common_dtype = np.dtype(np.result_type(np.array(dtype.type(0)), np.array(dtype_r.type(0))))
				common_dtypes.setdefault(dtype, dict())[dtype_r] = common_dtype
			except np.exceptions.DTypePromotionError:
				continue
			except ValueError:
				continue

	ufuncs = []
	for ufunc_name in all_features:
		ufunc: np.ufunc = getattr(np, ufunc_name)
		if not isinstance(ufunc, np.ufunc):
			raise ValueError(f"Not a ufunc: {ufunc_name}")

		casts = dict()
		specializations = dict()
		for type in ufunc.types:
			specialization = UFuncSpecialization.parse(type)
			if any(dtype not in supported_dtypes for dtype in [*specialization.input, specialization.output]):
				continue  # Skip unsupported dtype
			if len(set(specialization.input)) != 1:
				continue  # TODO Cannot specialize unequal type inputs right now.
			specializations[specialization.input] = specialization

		if ufunc.nin == 2:
			for l in supported_dtypes:
				for r in supported_dtypes:
					input = l, r
					if input in specializations:
						continue
					try:
						common_dtype = common_dtypes[l][r]
						casts[input] = specializations[(common_dtype, common_dtype)]
					except KeyError:
						print(f"Skipping cast {ufunc_name}({l.name}, {r.name}); not found.")

		ufuncs.append({
			"ufunc": ufunc_name,
			"specializations": [specialization.dumps() for specialization in specializations.values()],
			"casts": [f"{dump_dtypes(input)}->{model.dumps()}" for input, model in casts.items()]
		})

	with pathlib.Path("ufuncs.json").open("w") as f:
		json.dump({
			"ufuncs": ufuncs,
		}, f, indent='\t')

if __name__ == "__main__":
	main()
