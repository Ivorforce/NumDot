#ifndef VATENSOR_UFUNC_CONFIG_HPP
#define VATENSOR_UFUNC_CONFIG_HPP

#include "ufunc.hpp"
#include "util.hpp"
#include "vatensor/vcall.hpp"

namespace va {
	DEFINE_UFUNC_CALLER_UNARY(negative)
	DEFINE_UFUNC_CALLER_UNARY(sign)
	DEFINE_UFUNC_CALLER_UNARY(abs)
	DEFINE_UFUNC_CALLER_UNARY(square)
	DEFINE_UFUNC_CALLER_UNARY(sqrt)
	DEFINE_UFUNC_CALLER_UNARY(exp)
	DEFINE_UFUNC_CALLER_UNARY(log)
	DEFINE_UFUNC_CALLER_UNARY(rad2deg)
	DEFINE_UFUNC_CALLER_UNARY(deg2rad)

	DEFINE_UFUNC_CALLER_UNARY(conjugate)

	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(add)
	DEFINE_VFUNC_CALLER_BINARY(subtract)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(multiply)
	DEFINE_VFUNC_CALLER_BINARY(divide)
	DEFINE_VFUNC_CALLER_BINARY(remainder)
	DEFINE_VFUNC_CALLER_BINARY(pow)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(minimum)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(maximum)

	DEFINE_UFUNC_CALLER_UNARY(sin)
	DEFINE_UFUNC_CALLER_UNARY(cos)
	DEFINE_UFUNC_CALLER_UNARY(tan)
	DEFINE_UFUNC_CALLER_UNARY(asin)
	DEFINE_UFUNC_CALLER_UNARY(acos)
	DEFINE_UFUNC_CALLER_UNARY(atan)
	DEFINE_VFUNC_CALLER_BINARY(atan2)
	DEFINE_UFUNC_CALLER_UNARY(sinh)
	DEFINE_UFUNC_CALLER_UNARY(cosh)
	DEFINE_UFUNC_CALLER_UNARY(tanh)
	DEFINE_UFUNC_CALLER_UNARY(asinh)
	DEFINE_UFUNC_CALLER_UNARY(acosh)
	DEFINE_UFUNC_CALLER_UNARY(atanh)

	DEFINE_UFUNC_CALLER_UNARY(ceil)
	DEFINE_UFUNC_CALLER_UNARY(floor)
	DEFINE_UFUNC_CALLER_UNARY(trunc)
	DEFINE_UFUNC_CALLER_UNARY(round)
	DEFINE_UFUNC_CALLER_UNARY(rint)

	DEFINE_UFUNC_CALLER_UNARY(logical_not)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(logical_and)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(logical_or)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(logical_xor)

	DEFINE_UFUNC_CALLER_UNARY(bitwise_not)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(bitwise_and)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(bitwise_or)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(bitwise_xor)
	DEFINE_VFUNC_CALLER_BINARY(bitwise_left_shift)
	DEFINE_VFUNC_CALLER_BINARY(bitwise_right_shift)

	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(equal)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(not_equal)
	DEFINE_VFUNC_CALLER_BINARY(less)
	DEFINE_VFUNC_CALLER_BINARY(less_equal)
	DEFINE_VFUNC_CALLER_BINARY(greater)
	DEFINE_VFUNC_CALLER_BINARY(greater_equal)

	DEFINE_UFUNC_CALLER_UNARY(isnan)
	DEFINE_UFUNC_CALLER_UNARY(isfinite)
	DEFINE_UFUNC_CALLER_UNARY(isinf)
	DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE3(is_close, double, double, bool)
}

#endif //VATENSOR_UFUNC_CONFIG_HPP
