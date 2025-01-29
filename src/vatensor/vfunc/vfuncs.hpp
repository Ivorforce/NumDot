#ifndef VFUNCS_HPP
#define VFUNCS_HPP

#include "vatensor/varray.hpp"
#include "vatensor/vpromote.hpp"
#include "xtensor/xoperation.hpp"

#define IMPLEMENT_BINARY_VFUNC0(UFUNC_NAME, OP)\
template <typename R, typename A, typename B>\
inline void UFUNC_NAME(R& ret, const A& a, const B& b) {\
	va::broadcasting_assign_typesafe(ret, OP);\
}

#define IMPLEMENT_BINARY_VFUNC3(UFUNC_NAME, ARG1, ARG2, ARG3, OP)\
template <typename R, typename A, typename B>\
inline void UFUNC_NAME(R& ret, const A& a, const B& b, ARG1 arg1, ARG2 arg2, ARG3 arg3) {\
	va::broadcasting_assign_typesafe(ret, OP);\
}

#define IMPLEMENT_UNARY_VFUNC0(UFUNC_NAME, OP)\
template <typename R, typename A>\
inline void UFUNC_NAME(R& ret, const A& a) {\
	va::broadcasting_assign_typesafe(ret, OP);\
}

#define IMPLEMENT_UNARY_VFUNC1(UFUNC_NAME, ARG1, OP)\
template <typename R, typename A>\
inline void UFUNC_NAME(R& ret, const A& a, ARG1 arg1) {\
	va::broadcasting_assign_typesafe(ret, OP);\
}

#define BIT_SHIFT_SAFE(NAME, OP)\
struct NAME {\
	template <class T1, class T2>\
	constexpr std::decay_t<T1> operator()(T1&& arg1, T2&& arg2) const\
	{\
		constexpr std::decay_t<T2> bit_count = sizeof(arg1) * CHAR_BIT - std::is_signed<T1>::value;\
		return (arg2 < 0 || arg2 >= bit_count) ? 0 : (std::forward<T1>(arg1) OP std::forward<T2>(arg2));\
	}\
};

namespace va::op {
	BIT_SHIFT_SAFE(left_shift_safe, <<)
	BIT_SHIFT_SAFE(right_shift_safe, >>)

	struct bitwise_not_boolsafe {
		template <class T1>
		constexpr std::decay_t<T1> operator()(T1&& arg1) const { return ~arg1; }
		constexpr bool operator()(bool arg1) const { return !arg1; }
	};

	struct round_fun {
		template <class T1>
		constexpr std::decay_t<T1> operator()(T1&& arg1) const { return ::round(std::forward<T1>(arg1)); }

		template <class T1>
		constexpr std::decay_t<std::complex<T1>> operator()(const std::complex<T1>& arg1) const { return std::complex(::round(arg1.real()), ::round(arg1.imag())); }
	};

	struct conj_fun {
		// TODO Maybe these cases should just use identity (in the generator script).
		template <class T1>
		constexpr std::decay_t<T1> operator()(T1&& arg1) const { return std::forward<T1>(arg1); }

		template <class T1>
		constexpr std::decay_t<std::complex<T1>> operator()(const std::complex<T1>& arg1) const { return std::conj(arg1); }
	};

	template <class T1>
	constexpr auto sum(T1&& arg1, const va::axes_type* axes) {
		return axes
			? xt::sum(std::forward<T1>(arg1), *axes, xt::evaluation_strategy::lazy)
			: xt::sum(std::forward<T1>(arg1), xt::evaluation_strategy::lazy);
	}
}

namespace va::vfunc::impl {
	IMPLEMENT_UNARY_VFUNC0(negative, -va::promote::to_num(a))
	IMPLEMENT_UNARY_VFUNC0(sign, xt::sign(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(abs, xt::abs(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(square, xt::square(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(sqrt, xt::sqrt(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(exp, xt::exp(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(log, xt::log(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(rad2deg, xt::rad2deg(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(deg2rad, xt::deg2rad(va::promote::to_num(a)))

	IMPLEMENT_UNARY_VFUNC0(conjugate, xt::detail::make_xfunction<va::op::conj_fun>(va::promote::to_num(a)))

	// TODO calling xt::add directly etc. is broken because of the DEFINE_COMPLEX_OVERLOAD
	IMPLEMENT_BINARY_VFUNC0(add, xt::detail::make_xfunction<xt::detail::plus>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC0(subtract, xt::detail::make_xfunction<xt::detail::minus>(va::promote::to_num(a), va::promote::to_num(b)));
	IMPLEMENT_BINARY_VFUNC0(multiply, xt::detail::make_xfunction<xt::detail::multiplies>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC0(divide, xt::detail::make_xfunction<xt::detail::divides>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC0(remainder, xt::remainder(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC0(pow, xt::pow(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC0(minimum, xt::minimum(a, b))
	IMPLEMENT_BINARY_VFUNC0(maximum, xt::maximum(a, b))

	IMPLEMENT_UNARY_VFUNC1(sum, const va::axes_type*, va::op::sum(a, arg1))

	IMPLEMENT_UNARY_VFUNC0(sin, xt::sin(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(cos, xt::cos(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(tan, xt::tan(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(asin, xt::asin(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(acos, xt::acos(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(atan, xt::atan(va::promote::to_num(a)))
	IMPLEMENT_BINARY_VFUNC0(atan2, xt::atan2(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_UNARY_VFUNC0(sinh, xt::sinh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(cosh, xt::cosh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(tanh, xt::tanh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(asinh, xt::asinh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(acosh, xt::acosh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(atanh, xt::atanh(va::promote::to_num(a)))

	IMPLEMENT_UNARY_VFUNC0(ceil, xt::ceil(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(floor, xt::floor(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(trunc, xt::trunc(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(round, xt::detail::make_xfunction<va::op::round_fun>(a))
	// Actually uses nearbyint because rint can throw, which is undesirable in our case, and unlike numpy's behavior.
	IMPLEMENT_UNARY_VFUNC0(rint, xt::nearbyint(va::promote::to_num(a)))

	IMPLEMENT_UNARY_VFUNC0(logical_not, !va::promote::to_bool(a))
	// TODO RE-optimize these to short circuit on scalars
	IMPLEMENT_BINARY_VFUNC0(logical_and, va::promote::to_bool(a) && va::promote::to_bool(b))
	IMPLEMENT_BINARY_VFUNC0(logical_or, va::promote::to_bool(a) || va::promote::to_bool(b))
	IMPLEMENT_BINARY_VFUNC0(logical_xor, xt::detail::make_xfunction<xt::detail::not_equal_to>(va::promote::to_bool(a), va::promote::to_bool(b)))

	IMPLEMENT_UNARY_VFUNC0(bitwise_not, xt::detail::make_xfunction<va::op::bitwise_not_boolsafe>(a))
	IMPLEMENT_BINARY_VFUNC0(bitwise_and, a & b)
	IMPLEMENT_BINARY_VFUNC0(bitwise_or, a | b)
	IMPLEMENT_BINARY_VFUNC0(bitwise_xor, a ^ b)
	IMPLEMENT_BINARY_VFUNC0(bitwise_left_shift, xt::detail::make_xfunction<va::op::left_shift_safe>(a, b))
	IMPLEMENT_BINARY_VFUNC0(bitwise_right_shift, xt::detail::make_xfunction<va::op::right_shift_safe>(a, b))

	IMPLEMENT_BINARY_VFUNC0(equal, xt::equal(a, b))
	IMPLEMENT_BINARY_VFUNC0(not_equal, xt::not_equal(a, b))
	IMPLEMENT_BINARY_VFUNC0(less, xt::less(a, b))
	IMPLEMENT_BINARY_VFUNC0(less_equal, xt::less_equal(a, b))
	IMPLEMENT_BINARY_VFUNC0(greater, xt::greater(a, b))
	IMPLEMENT_BINARY_VFUNC0(greater_equal, xt::greater_equal(a, b))

	IMPLEMENT_UNARY_VFUNC0(isnan, xt::isnan(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(isfinite, xt::isfinite(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC0(isinf, xt::isinf(va::promote::to_num(a)))
	IMPLEMENT_BINARY_VFUNC3(is_close, double, double, bool, xt::isclose(va::promote::to_num(a), arg1, arg2, arg3))
} // namespace va::vfunc::impl

#endif //VFUNCS_HPP
