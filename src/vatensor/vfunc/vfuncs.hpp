#ifndef VFUNCS_HPP
#define VFUNCS_HPP

#include <xtensor-signal/fft.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xsort.hpp>

#include "vatensor/varray.hpp"
#include "vatensor/vpromote.hpp"
#include "vatensor/vassign.hpp"
#include "xtensor/xoperation.hpp"

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

	template <typename R, typename I>
	R va_cast(I i) {
		return static_cast<R>(i);
	}

	std::complex<float_t> va_cast(std::complex<double_t> i) {
		return static_cast<std::complex<float_t>>(i);
	}

	std::complex<double_t> va_cast(std::complex<float_t> i) {
		return static_cast<std::complex<float_t>>(i);
	}

	// FIXME These don't support axes yet, see https://github.com/xtensor-stack/xtensor/issues/1555
	using namespace xt;
	XTENSOR_REDUCER_FUNCTION(va_any, xt::detail::logical_or, bool, true)
	XTENSOR_REDUCER_FUNCTION(va_all, xt::detail::logical_and, bool, false)

	template <class T1>
	auto median(const T1 &t1, const axes_type& axes) {
		// TODO Improve implementation, this one is very finnicky
		if (axes.size() > 1) {
			throw std::runtime_error("only single axis median is supported right now");
		}
		return xt::median(t1, axes[0]);
	}

	template <class VWrite, class I>
	auto vcast(const I& cvalue) {
		using VRead = typename I::value_type;

		if constexpr (std::is_same_v<VWrite, bool> && xtl::is_complex<VRead>::value) {
			// This helps mostly complex dtypes to booleanize
			return xt::cast<uint8_t>(xt::equal(cvalue, static_cast<VRead>(0)));
		}
		else if constexpr (!std::is_convertible_v<VRead, VWrite>) {
			throw std::runtime_error("Cannot promote in this way.");
			return xt::xscalar<VWrite>();  // To give us a return type.
		}
#ifdef XTENSOR_USE_XSIMD
		// For some reason, bool - to - bool assignments are broken in xsimd
		// TODO Should make this reproducible, I haven't managed so far.
		// See https://github.com/Ivorforce/NumDot/issues/123
		else if constexpr (std::is_same_v<VWrite, bool> && std::is_same_v<VRead, bool>) {
			return xt::cast<uint8_t>(cvalue);
		}
		else if constexpr (xtl::is_complex<VWrite>::value) {
			// xsimd also has no auto conversion into complex types
			return xt::cast<VWrite>(cvalue);
		}
#endif
		else
		{
			return cvalue;
		}
	}
}

#define IMPLEMENT_INPLACE_VFUNC(UFUNC_NAME, OP, ...)\
template <typename R>\
inline void UFUNC_NAME(R& ret, ##__VA_ARGS__) {\
	va::broadcasting_assign_typesafe(ret, OP);\
}

#define IMPLEMENT_UNARY_VFUNC(UFUNC_NAME, OP, ...)\
template <typename R, typename A>\
inline void UFUNC_NAME(R& ret, const A& a, ##__VA_ARGS__) {\
	va::broadcasting_assign_typesafe(ret, OP);\
}

#define IMPLEMENT_BINARY_VFUNC(UFUNC_NAME, OP, ...)\
template <typename R, typename A, typename B>\
inline void UFUNC_NAME(R& ret, const A& a, const B& b, ##__VA_ARGS__) {\
	va::broadcasting_assign_typesafe(ret, OP);\
}

#define IMPLEMENT_UNARY_RFUNC(UFUNC_NAME, SINGLE, MULTI)\
template <typename R, typename A>\
inline void UFUNC_NAME(R& ret, const A& a, const va::axes_type* axes) {\
	if (axes) {\
		va::broadcasting_assign_typesafe(ret, MULTI);\
	}\
	else {\
		const typename R::value_type intermediate = va::op::va_cast<typename R::value_type>(SINGLE);\
		broadcasting_assign_typesafe(ret, xt::xscalar<typename R::value_type>(intermediate));\
	}\
}

#define IMPLEMENT_BINARY_RFUNC(UFUNC_NAME, SINGLE, MULTI)\
template <typename R, typename A, typename B>\
inline void UFUNC_NAME(R& ret, const A& a, const B& b, const va::axes_type* axes) {\
	if (axes) {\
		va::broadcasting_assign_typesafe(ret, MULTI);\
	}\
	else {\
		const typename R::value_type intermediate = va::op::va_cast<typename R::value_type>(SINGLE);\
		broadcasting_assign_typesafe(ret, xt::xscalar<typename R::value_type>(intermediate));\
	}\
}

namespace va::vfunc::impl {
	IMPLEMENT_INPLACE_VFUNC(fill, xt::xscalar(reinterpret_cast<typename R::value_type&>(fill_value)), void* fill_value)
	IMPLEMENT_UNARY_VFUNC(assign, va::op::vcast<typename R::value_type>(a))

	IMPLEMENT_UNARY_VFUNC(negative, -va::promote::to_num(a))
	IMPLEMENT_UNARY_VFUNC(sign, xt::sign(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(abs, xt::abs(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(square, xt::square(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(sqrt, xt::sqrt(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(exp, xt::exp(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(log, xt::log(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(rad2deg, xt::rad2deg(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(deg2rad, xt::deg2rad(va::promote::to_num(a)))

	IMPLEMENT_UNARY_VFUNC(conjugate, xt::detail::make_xfunction<va::op::conj_fun>(va::promote::to_num(a)))

	// TODO calling xt::add directly etc. is broken because of the DEFINE_COMPLEX_OVERLOAD
	IMPLEMENT_BINARY_VFUNC(add, xt::detail::make_xfunction<xt::detail::plus>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(subtract, xt::detail::make_xfunction<xt::detail::minus>(va::promote::to_num(a), va::promote::to_num(b)));
	IMPLEMENT_BINARY_VFUNC(multiply, xt::detail::make_xfunction<xt::detail::multiplies>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(divide, xt::detail::make_xfunction<xt::detail::divides>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(remainder, xt::remainder(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(pow, xt::pow(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(minimum, xt::minimum(a, b))
	IMPLEMENT_BINARY_VFUNC(maximum, xt::maximum(a, b))

	IMPLEMENT_UNARY_RFUNC(sum, xt::sum(a)(), xt::sum(a, *axes))
	IMPLEMENT_UNARY_RFUNC(prod, xt::prod(a)(), xt::prod(a, *axes))
	IMPLEMENT_UNARY_RFUNC(mean, xt::mean(a)(), xt::mean(a, *axes))
	IMPLEMENT_UNARY_RFUNC(median, xt::median(a), va::op::median(a, *axes))
	IMPLEMENT_UNARY_RFUNC(variance, xt::variance(a)(), xt::variance(a, *axes))
	IMPLEMENT_UNARY_RFUNC(standard_deviation, xt::stddev(a)(), xt::stddev(a, *axes))
	IMPLEMENT_UNARY_RFUNC(max, xt::amax(a)(), xt::amax(a, *axes))
	IMPLEMENT_UNARY_RFUNC(min, xt::amin(a)(), xt::amin(a, *axes))
	IMPLEMENT_UNARY_RFUNC(norm_l0, xt::norm_l0(a)(), xt::norm_l0(a, *axes, xt::evaluation_strategy::lazy))
	IMPLEMENT_UNARY_RFUNC(norm_l1, xt::norm_l1(a)(), xt::norm_l1(a, *axes, xt::evaluation_strategy::lazy))
	IMPLEMENT_UNARY_RFUNC(norm_l2, xt::norm_l2(a)(), xt::norm_l2(a, *axes, xt::evaluation_strategy::lazy))
	IMPLEMENT_UNARY_RFUNC(norm_linf, xt::norm_linf(a)(), xt::norm_linf(a, *axes, xt::evaluation_strategy::lazy))

	IMPLEMENT_UNARY_RFUNC(all, va::op::va_all(a)(), va::op::va_all(a, *axes))
	IMPLEMENT_UNARY_RFUNC(any, va::op::va_any(a)(), va::op::va_any(a, *axes))

	IMPLEMENT_UNARY_VFUNC(sin, xt::sin(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(cos, xt::cos(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(tan, xt::tan(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(asin, xt::asin(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(acos, xt::acos(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(atan, xt::atan(va::promote::to_num(a)))
	IMPLEMENT_BINARY_VFUNC(atan2, xt::atan2(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_UNARY_VFUNC(sinh, xt::sinh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(cosh, xt::cosh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(tanh, xt::tanh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(asinh, xt::asinh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(acosh, xt::acosh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(atanh, xt::atanh(va::promote::to_num(a)))

	IMPLEMENT_UNARY_VFUNC(ceil, xt::ceil(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(floor, xt::floor(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(trunc, xt::trunc(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(round, xt::detail::make_xfunction<va::op::round_fun>(a))
	// Actually uses nearbyint because rint can throw, which is undesirable in our case, and unlike numpy's behavior.
	IMPLEMENT_UNARY_VFUNC(rint, xt::nearbyint(va::promote::to_num(a)))

	IMPLEMENT_UNARY_VFUNC(logical_not, !va::promote::to_bool(a))
	// TODO RE-optimize these to short circuit on scalars
	IMPLEMENT_BINARY_VFUNC(logical_and, va::promote::to_bool(a) && va::promote::to_bool(b))
	IMPLEMENT_BINARY_VFUNC(logical_or, va::promote::to_bool(a) || va::promote::to_bool(b))
	IMPLEMENT_BINARY_VFUNC(logical_xor, xt::detail::make_xfunction<xt::detail::not_equal_to>(va::promote::to_bool(a), va::promote::to_bool(b)))

	IMPLEMENT_UNARY_VFUNC(bitwise_not, xt::detail::make_xfunction<va::op::bitwise_not_boolsafe>(a))
	IMPLEMENT_BINARY_VFUNC(bitwise_and, a & b)
	IMPLEMENT_BINARY_VFUNC(bitwise_or, a | b)
	IMPLEMENT_BINARY_VFUNC(bitwise_xor, a ^ b)
	IMPLEMENT_BINARY_VFUNC(bitwise_left_shift, xt::detail::make_xfunction<va::op::left_shift_safe>(a, b))
	IMPLEMENT_BINARY_VFUNC(bitwise_right_shift, xt::detail::make_xfunction<va::op::right_shift_safe>(a, b))

	IMPLEMENT_BINARY_VFUNC(equal, xt::equal(a, b))
	IMPLEMENT_BINARY_VFUNC(not_equal, xt::not_equal(a, b))
	IMPLEMENT_BINARY_VFUNC(less, xt::less(a, b))
	IMPLEMENT_BINARY_VFUNC(less_equal, xt::less_equal(a, b))
	IMPLEMENT_BINARY_VFUNC(greater, xt::greater(a, b))
	IMPLEMENT_BINARY_VFUNC(greater_equal, xt::greater_equal(a, b))

	IMPLEMENT_UNARY_VFUNC(isnan, xt::isnan(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(isfinite, xt::isfinite(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(isinf, xt::isinf(va::promote::to_num(a)))

	IMPLEMENT_BINARY_VFUNC(is_close, xt::isclose(va::promote::to_num(a), va::promote::to_num(b), rtol, atol, equal_nan), double rtol, double atol, bool equal_nan)
	IMPLEMENT_BINARY_VFUNC(array_equiv, xt::xscalar<bool>(xt::all(xt::equal(a, b))))
	IMPLEMENT_BINARY_VFUNC(all_close, xt::xscalar<bool>(xt::all(xt::isclose(va::promote::to_num(a), va::promote::to_num(b), rtol, atol, equal_nan))), double rtol, double atol, bool equal_nan)

	IMPLEMENT_UNARY_VFUNC(fft, xt::fft::fft(std::forward<decltype(a)>(a), axis), std::ptrdiff_t axis)
	IMPLEMENT_UNARY_VFUNC(
		pad,
		xt::pad(std::forward<decltype(a)>(a), pad_width, pad_mode, reinterpret_cast<typename A::value_type&>(pad_value)),
		std::vector<std::vector<std::size_t>>& pad_width,
		xt::pad_mode pad_mode,
		void* pad_value
	)

	IMPLEMENT_BINARY_RFUNC(reduce_dot, xt::sum(a * b)(), xt::sum(a * b, *axes))

	IMPLEMENT_BINARY_VFUNC(a0xb1_minus_a1xb0,
		(xt::strided_view(a, { xt::ellipsis(), i0 }) * xt::strided_view(b, { xt::ellipsis(), i1 }))
		- (xt::strided_view(a, { xt::ellipsis(), i1 }) * xt::strided_view(b, { xt::ellipsis(), i0 })),
		const std::ptrdiff_t i0, const std::ptrdiff_t i1
	)
} // namespace va::vfunc::impl

#endif //VFUNCS_HPP
