"""Differential test: every NDArray.assign_* method must produce the same
result as its nd.* counterpart, written into a preallocated target.

Ground truth is nd.* itself (whose correctness is covered by the array-api
conformance suite), so comparisons are bit-exact — both paths run the same
C++ kernels and differ only in target plumbing.
"""

import numpy as np
import pytest

from .client import BridgeClient


# ---- deterministic inputs ----

F = np.array([
	[0.25, -0.5, 0.75, 1.25],
	[2.0, -3.5, 0.125, 4.0],
	[-1.0, 0.5, 2.5, -0.75],
])
F2 = np.array([
	[1.5, 2.5, -0.25, 0.5],
	[-2.0, 0.75, 3.0, -1.25],
	[0.5, -0.125, 1.0, 2.0],
])
FPOS = np.abs(F) + 0.25       # > 0 (log, sqrt, divide, remainder)
FUNIT = F / 5.0               # in (-1, 1) (asin, acos, atanh)
FGE1 = FPOS + 1.0             # >= 1 (acosh)
I = np.array([[12, 10, 25, 7], [3, 255, 15, 1], [255, 128, 64, 5]], dtype=np.int32)
I2 = np.array([[10, 6, 13, 255], [1, 60, 240, 2], [7, 255, 33, 9]], dtype=np.int32)
SH = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [1, 2, 3, 0]], dtype=np.int32)
B = np.array([[True, False, True, True], [False, False, True, False], [True, True, False, False]])
B2 = np.array([[True, True, False, True], [False, True, False, False], [True, False, True, False]])
C = np.array([[1 + 2j, -3 + 0.5j], [0 - 1j, 2.5 + 2.5j]], dtype=np.complex128)
M45 = np.arange(20, dtype=np.float64).reshape(4, 5) / 4.0 - 2.0
V3 = np.array([1.5, -2.0, 3.25])
V3B = np.array([-0.5, 4.0, 2.0])
A23 = np.array([[1.0, -2.0, 3.0], [0.5, 2.5, -1.5]])
A23B = np.array([[-3.0, 1.0, 0.5], [2.0, -0.25, 4.0]])
V8 = np.linspace(-2.0, 3.0, 8)
K3 = np.array([0.25, 0.5, 0.25])

# (assign method suffix, nd counterpart, args). Every assign_* method on
# NDArray must appear here exactly once — enforced by
# test_all_assign_methods_covered.
CASES = [
	# unary elementwise
	("abs", "abs", (F,)),
	("acos", "acos", (FUNIT,)),
	("acosh", "acosh", (FGE1,)),
	("asin", "asin", (FUNIT,)),
	("asinh", "asinh", (F,)),
	("atan", "atan", (F,)),
	("atanh", "atanh", (FUNIT,)),
	("ceil", "ceil", (F,)),
	("cos", "cos", (F,)),
	("cosh", "cosh", (F,)),
	("deg2rad", "deg2rad", (F,)),
	("exp", "exp", (F,)),
	("floor", "floor", (F,)),
	("log", "log", (FPOS,)),
	("negative", "negative", (F,)),
	("positive", "positive", (F,)),
	("rad2deg", "rad2deg", (F,)),
	("rint", "rint", (F,)),
	("round", "round", (F,)),
	("sign", "sign", (F,)),
	("sin", "sin", (F,)),
	("sinh", "sinh", (F,)),
	("sqrt", "sqrt", (FPOS,)),
	("square", "square", (F,)),
	("tan", "tan", (F,)),
	("tanh", "tanh", (F,)),
	("trunc", "trunc", (F,)),
	("angle", "angle", (C,)),
	("conjugate", "conjugate", (C,)),
	("logical_not", "logical_not", (B,)),
	("bitwise_not", "bitwise_not", (I,)),
	# binary elementwise
	("add", "add", (F, F2)),
	("subtract", "subtract", (F, F2)),
	("multiply", "multiply", (F, F2)),
	("divide", "divide", (F, FPOS)),
	("pow", "pow", (FPOS, F2)),
	("remainder", "remainder", (F, FPOS)),
	("maximum", "maximum", (F, F2)),
	("minimum", "minimum", (F, F2)),
	("atan2", "atan2", (F, F2)),
	("equal", "equal", (F, F2)),
	("not_equal", "not_equal", (F, F2)),
	("greater", "greater", (F, F2)),
	("greater_equal", "greater_equal", (F, F2)),
	("less", "less", (F, F2)),
	("less_equal", "less_equal", (F, F2)),
	("logical_and", "logical_and", (B, B2)),
	("logical_or", "logical_or", (B, B2)),
	("logical_xor", "logical_xor", (B, B2)),
	("bitwise_and", "bitwise_and", (I, I2)),
	("bitwise_or", "bitwise_or", (I, I2)),
	("bitwise_xor", "bitwise_xor", (I, I2)),
	("bitwise_left_shift", "bitwise_left_shift", (I, SH)),
	("bitwise_right_shift", "bitwise_right_shift", (I, SH)),
	# other
	("clip", "clip", (F, 0.0, 1.0)),
	("matmul", "matmul", (F, M45)),
	("dot", "dot", (V3, V3B)),
	("cross", "cross", (A23, A23B)),
	("convolve", "convolve", (V8, K3)),
	# reductions
	("sum", "sum", (F, 1)),
	("prod", "prod", (F, 1)),
	("mean", "mean", (F, 1)),
	("std", "std", (F, 1)),
	("var", "var", (F, 1)),
	("max", "max", (F, 1)),
	("min", "min", (F, 1)),
	("all", "all", (B, 1)),
	("any", "any", (B, 1)),
	("norm", "norm", (F, 2, 1)),
	("reduce_dot", "sum_product", (F, F2, 1)),
]


@pytest.fixture(scope="module")
def assign_bridge(godot_binary, tmp_path_factory):
	# One Godot process for the whole module — spawning per parametrized case
	# would dominate the runtime.
	log_path = tmp_path_factory.mktemp("assign_parity") / "godot.log"
	with BridgeClient(godot_binary=godot_binary, log_path=log_path) as client:
		yield client


def assert_bit_equal(out: np.ndarray, expected: np.ndarray) -> None:
	assert out.dtype == expected.dtype
	assert out.shape == expected.shape
	assert out.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
	"assign_suffix,nd_func,args",
	CASES,
	ids=[case[0] for case in CASES],
)
def test_assign_matches_nd(assign_bridge, assign_suffix, nd_func, args):
	expected = assign_bridge.call_nd(nd_func, *args)
	target = np.zeros(expected.shape, expected.dtype)
	out = assign_bridge.call_assign(f"assign_{assign_suffix}", target, *args)
	assert_bit_equal(out, expected)


def test_all_assign_methods_covered(assign_bridge):
	header, _ = assign_bridge.call("list_assign_methods")
	assert header.get("ok")
	methods = set(header["methods"])
	cased = {f"assign_{suffix}" for suffix, _, _ in CASES}
	assert methods - cased == set(), f"assign_* methods without a parity case: {sorted(methods - cased)}"
	assert cased - methods == set(), f"parity cases without an assign_* method: {sorted(cased - methods)}"


# ---- dtype-mismatched targets (the temp-then-cast path in evaluate_target) ----

CAST_CASES = [
	# The exact #180 scenario: uint8 sum promotes to uint64 internally, then
	# casts back into the uint8 target.
	("sum", "sum", (np.ones((8, 8, 4), dtype=np.uint8), 2), np.uint8),
	("add", "add", (F, F2), np.float32),
	("mean", "mean", (F, 1), np.float32),
]


@pytest.mark.parametrize(
	"assign_suffix,nd_func,args,target_dtype",
	CAST_CASES,
	ids=[f"{case[0]}-into-{np.dtype(case[3]).name}" for case in CAST_CASES],
)
def test_assign_casts_into_target_dtype(assign_bridge, assign_suffix, nd_func, args, target_dtype):
	expected = assign_bridge.call_nd(nd_func, *args).astype(target_dtype)
	target = np.zeros(expected.shape, target_dtype)
	out = assign_bridge.call_assign(f"assign_{assign_suffix}", target, *args)
	assert_bit_equal(out, expected)


# ---- error paths ----

def test_assign_wrong_shape_target_errors_cleanly(assign_bridge):
	# assign_* only prints on error and leaves the target untouched, so the
	# observable contract is: no crash, target unchanged, bridge alive.
	tensor = np.ones((8, 8, 4), dtype=np.uint8)
	target = np.zeros((3, 3), dtype=np.uint64)
	out = assign_bridge.call_assign("assign_sum", target, tensor, 2)
	assert np.all(out == 0)
	header, _ = assign_bridge.call("ping")
	assert header == {"ok": True, "pong": True}
