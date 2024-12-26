#ifndef VATENSOR_UFUNC_CONFIG_HPP
#define VATENSOR_UFUNC_CONFIG_HPP

#include "ufunc.hpp"
#include "util.hpp"
#include "vatensor/vcall.hpp"
#include "vatensor/vpromote.hpp"
#include "xtensor/xoperation.hpp"

#define BIT_SHIFT_SAFE(NAME, OP)\
struct NAME {\
	template <class T1, class T2>\
	constexpr std::decay_t<T1> operator()(T1&& arg1, T2&& arg2) const\
	{\
		constexpr std::decay_t<T2> bit_count = sizeof(arg1) * CHAR_BIT;\
		return (arg2 >= bit_count) ? 0 : (std::forward<T1>(arg1) OP std::forward<T2>(arg2));\
	}\
};

namespace va::op {
	BIT_SHIFT_SAFE(left_shift_safe, <<)
	BIT_SHIFT_SAFE(right_shift_safe, >>)
}

UNARY_UFUNC(negative, -va::promote::to_num(a))
UNARY_UFUNC(sign, xt::sign(va::promote::to_num(a)))
UNARY_UFUNC(abs, xt::abs(va::promote::to_num(a)))
UNARY_UFUNC(square, xt::square(va::promote::to_num(a)))
UNARY_UFUNC(sqrt, xt::sqrt(va::promote::to_num(a)))
UNARY_UFUNC(exp, xt::exp(va::promote::to_num(a)))
UNARY_UFUNC(log, xt::log(va::promote::to_num(a)))
UNARY_UFUNC(rad2deg, xt::rad2deg(va::promote::to_num(a)))
UNARY_UFUNC(deg2rad, xt::deg2rad(va::promote::to_num(a)))
// TODO calling xt::add directly etc. is broken because of the DEFINE_COMPLEX_OVERLOAD
BINARY_UFUNC(add, xt::detail::make_xfunction<xt::detail::plus>(va::promote::to_num(a), va::promote::to_num(b)))
BINARY_CALLER_COMMUTATIVE(add)
BINARY_UFUNC(subtract, xt::detail::make_xfunction<xt::detail::minus>(va::promote::to_num(a), va::promote::to_num(b)))
BINARY_CALLER(subtract)
BINARY_UFUNC(multiply, xt::detail::make_xfunction<xt::detail::multiplies>(va::promote::to_num(a), va::promote::to_num(b)))
BINARY_CALLER_COMMUTATIVE(multiply)
BINARY_UFUNC(divide, xt::detail::make_xfunction<xt::detail::divides>(va::promote::to_num(a), va::promote::to_num(b)))
BINARY_CALLER(divide)
BINARY_UFUNC(remainder, xt::remainder(va::promote::to_num(a), va::promote::to_num(b)))
BINARY_CALLER(remainder)
BINARY_UFUNC(pow, xt::pow(va::promote::to_num(a), va::promote::to_num(b)))
BINARY_CALLER(pow)
BINARY_UFUNC(minimum, xt::minimum(a, b))
BINARY_CALLER_COMMUTATIVE(minimum)
BINARY_UFUNC(maximum, xt::maximum(a, b))
BINARY_CALLER_COMMUTATIVE(maximum)

UNARY_UFUNC(sin, xt::sin(va::promote::to_num(a)))
UNARY_UFUNC(cos, xt::cos(va::promote::to_num(a)))
UNARY_UFUNC(tan, xt::tan(va::promote::to_num(a)))
UNARY_UFUNC(asin, xt::asin(va::promote::to_num(a)))
UNARY_UFUNC(acos, xt::acos(va::promote::to_num(a)))
UNARY_UFUNC(atan, xt::atan(va::promote::to_num(a)))
BINARY_UFUNC(atan2, xt::atan2(va::promote::to_num(a), va::promote::to_num(b)))
BINARY_CALLER(atan2)
UNARY_UFUNC(sinh, xt::sinh(va::promote::to_num(a)))
UNARY_UFUNC(cosh, xt::cosh(va::promote::to_num(a)))
UNARY_UFUNC(tanh, xt::tanh(va::promote::to_num(a)))
UNARY_UFUNC(asinh, xt::asinh(va::promote::to_num(a)))
UNARY_UFUNC(acosh, xt::acosh(va::promote::to_num(a)))
UNARY_UFUNC(atanh, xt::atanh(va::promote::to_num(a)))

UNARY_UFUNC(ceil, xt::ceil(va::promote::to_num(a)))
UNARY_UFUNC(floor, xt::floor(va::promote::to_num(a)))
UNARY_UFUNC(trunc, xt::trunc(va::promote::to_num(a)))
// UNARY_UFUNC(round, xt::round(va::promote::to_num(a)))
// Actually uses nearbyint because rint can throw, which is undesirable in our case, and unlike numpy's behavior.
UNARY_UFUNC(rint, xt::nearbyint(va::promote::to_num(a)))

UNARY_UFUNC(logical_not, !a)
// TODO RE-optimize these to short circuit on scalars
BINARY_UFUNC(logical_and, a && b)
BINARY_CALLER_COMMUTATIVE(logical_and)
BINARY_UFUNC(logical_or, a || b)
BINARY_CALLER_COMMUTATIVE(logical_or)
BINARY_UFUNC(logical_xor, xt::detail::make_xfunction<xt::detail::not_equal_to>(a, b))
BINARY_CALLER_COMMUTATIVE(logical_xor)

UNARY_UFUNC(bitwise_not, ~a)
BINARY_UFUNC(bitwise_and, a & b)
BINARY_CALLER_COMMUTATIVE(bitwise_and)
BINARY_UFUNC(bitwise_or, a | b)
BINARY_CALLER_COMMUTATIVE(bitwise_or)
BINARY_UFUNC(bitwise_xor, a ^ b)
BINARY_CALLER_COMMUTATIVE(bitwise_xor)
BINARY_UFUNC(bitwise_left_shift, xt::detail::make_xfunction<va::op::left_shift_safe>(a, b))
BINARY_CALLER(bitwise_left_shift)
BINARY_UFUNC(bitwise_right_shift, xt::detail::make_xfunction<va::op::right_shift_safe>(a, b))
BINARY_CALLER(bitwise_right_shift)

BINARY_UFUNC(equal, xt::equal(a, b))
BINARY_CALLER_COMMUTATIVE(equal)
BINARY_UFUNC(not_equal, xt::not_equal(a, b))
BINARY_CALLER_COMMUTATIVE(not_equal)
BINARY_UFUNC(less, xt::less(a, b))
BINARY_CALLER(less)
BINARY_UFUNC(less_equal, xt::less_equal(a, b))
BINARY_CALLER(less_equal)
BINARY_UFUNC(greater, xt::greater(a, b))
BINARY_CALLER(greater)
BINARY_UFUNC(greater_equal, xt::greater_equal(a, b))
BINARY_CALLER(greater_equal)
UNARY_UFUNC(isnan, xt::isnan(va::promote::to_num(a)))
UNARY_UFUNC(isfinite, xt::isfinite(va::promote::to_num(a)))
UNARY_UFUNC(isinf, xt::isinf(va::promote::to_num(a)))

#endif //VATENSOR_UFUNC_CONFIG_HPP
