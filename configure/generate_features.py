import dataclasses
import json
import pathlib

import numpy as np
from typing_extensions import OrderedDict

# Should be set but must be ordered.
supported_dtypes: list[np.dtype] = [
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
]

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

	"conjugate",

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

	vfuncs = []
	for ufunc_name in all_features:
		ufunc: np.ufunc = getattr(np, ufunc_name)
		if not isinstance(ufunc, np.ufunc):
			raise ValueError(f"Not a ufunc: {ufunc_name}")

		casts = OrderedDict()
		specializations = OrderedDict()
		for type in sorted(ufunc.types):
			# FIXME Not sure why but at some point, NumPy started using long double instead of double for many normal operations (probably v2).
			# We do not support long double yet.
			specialization = UFuncSpecialization.parse(type.replace("g", "d").replace("G", "D"))
			if any(dtype not in supported_dtypes for dtype in [*specialization.input, specialization.output]):
				continue  # Skip unsupported dtype
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

		vfuncs.append({
			"name": ufunc_name,
			"specializations": [specialization.dumps() for specialization in specializations.values()],
			"casts": [f"{dump_dtypes(input)}->{model.dumps()}" for input, model in casts.items()],
		})

	# Not a ufunc because (in NumPy), it accepts a 'decimals' int parameter.
	vfuncs.append({
		"name": "round",
		"specializations": ["f->f", "d->d", "F->F", "D->D"],
		"casts": [],
	})
	vfuncs.append({
		"name": "is_close",
		# TODO Could implement other dtypes by just calling equals in the implementation.
		"specializations": ["ff->b", "dd->b", "FF->b", "DD->b"],
		"casts": [],
		"vargs": ["double", "double", "bool"]
	})
	vfuncs.append({
		"name": "sum",
		# TODO Check if these specializations make sense.
		"specializations": [
			"?->?",
			"B->B",
			"D->D",
			"F->F",
			"H->H",
			"I->I",
			"Q->Q",
			"b->b",
			"d->d",
			"f->f",
			"h->h",
			"i->i",
			"q->q"
		],
		"casts": [],
		"vargs": ["const va::axes_type*"]
	})

	with (pathlib.Path(__file__).parent / "vfuncs.json").open("w") as f:
		json.dump({
			"vfuncs": vfuncs,
		}, f, indent='\t')

if __name__ == "__main__":
	main()
