#ifndef VFUNCS_HPP
#define VFUNCS_HPP

#include <xtensor-signal/fft.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/misc/xpad.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/misc/xsort.hpp>

#include "vatensor/varray.hpp"
#include "vatensor/vpromote.hpp"
#include "vatensor/vassign.hpp"
#include "xtensor/core/xoperation.hpp"
#include "godot_cpp/core/error_macros.hpp"  // for WARN_PRINT_ONCE

namespace va::op {
	// Spec (array-api): if x2 >= bit_count, the result is 0. Shifting into the
	// sign bit is UB in C++, so for signed types the safe range stops one short.
	struct left_shift_safe {
		template <class T1, class T2>
		constexpr std::decay_t<T1> operator()(T1&& arg1, T2&& arg2) const {
			constexpr std::decay_t<T2> bit_count = sizeof(arg1) * CHAR_BIT - std::is_signed<T1>::value;
			return (arg2 < 0 || arg2 >= bit_count) ? 0 : (std::forward<T1>(arg1) << std::forward<T2>(arg2));
		}
	};

	// Spec (array-api): if x2 >= bit_count, the result is 0 for non-negative x1
	// and -1 for negative x1 (sign-extending arithmetic shift). The previous
	// unified BIT_SHIFT_SAFE always returned 0 — wrong for negative inputs.
	struct right_shift_safe {
		template <class T1, class T2>
		constexpr std::decay_t<T1> operator()(T1&& arg1, T2&& arg2) const {
			using R = std::decay_t<T1>;
			constexpr std::decay_t<T2> bit_count = sizeof(arg1) * CHAR_BIT;
			if (arg2 < 0) return R(0);
			if (arg2 >= bit_count) {
				if constexpr (std::is_signed_v<R>) return arg1 < 0 ? R(-1) : R(0);
				else return R(0);
			}
			return std::forward<T1>(arg1) >> std::forward<T2>(arg2);
		}
	};

	struct bitwise_not_boolsafe {
		template <class T1>
		constexpr std::decay_t<T1> operator()(T1&& arg1) const { return ~arg1; }
		constexpr bool operator()(bool arg1) const { return !arg1; }
	};

	struct round_fun {
		template <class T1>
		constexpr std::decay_t<T1> operator()(T1&& arg1) const { return ::round(std::forward<T1>(arg1)); }

		template <class T1>
		constexpr std::decay_t<std::complex<T1>> operator()(const std::complex<T1>& arg1) const { return std::complex<T1>(::round(arg1.real()), ::round(arg1.imag())); }
	};

	struct conj_fun {
		// TODO Maybe these cases should just use identity (in the generator script).
		template <class T1>
		constexpr std::decay_t<T1> operator()(T1&& arg1) const { return std::forward<T1>(arg1); }

		template <class T1>
		constexpr std::decay_t<std::complex<T1>> operator()(const std::complex<T1>& arg1) const { return std::conj(arg1); }
	};

	struct signbit_fun {
		template <class T1>
		constexpr bool operator()(T1 arg1) const { return std::signbit(arg1); }
	};

	struct copysign_fun {
		template <class T1, class T2>
		constexpr std::decay_t<T1> operator()(T1 arg1, T2 arg2) const { return std::copysign(arg1, static_cast<T1>(arg2)); }
	};

	struct floor_divide_fun {
		template <class T1, class T2>
		constexpr auto operator()(T1 a, T2 b) const {
			using R = std::common_type_t<T1, T2>;
			if constexpr (std::is_integral_v<R>) {
				const R aa = static_cast<R>(a);
				const R bb = static_cast<R>(b);
				const R q = aa / bb;
				const R r = aa - q * bb;
				return (r != 0 && (r < 0) != (bb < 0)) ? R(q - 1) : q;
			}
			else {
				return std::floor(static_cast<R>(a) / static_cast<R>(b));
			}
		}
	};

	// std::complex's operator/ computes (ac+bd)/(c²+d²) directly, so the
	// denominator overflows when |c|² + |d|² exceeds F's max even though the
	// quotient is finite (e.g. complex64 with |c| ≈ 1.8e19). Smith's algorithm
	// scales by min(|c|,|d|)/max(|c|,|d|) first so neither intermediate squares.
	// Numpy's complex divide uses the same trick.
	struct divide_fun {
		template <class T1, class T2>
		constexpr auto operator()(const T1& a, const T2& b) const {
			using R = std::common_type_t<T1, T2>;
			if constexpr (xtl::is_complex<R>::value) {
				using F = typename R::value_type;
				const R z1 = static_cast<R>(a);
				const R z2 = static_cast<R>(b);
				const F c = z2.real();
				const F d = z2.imag();
				const F p = z1.real();
				const F q = z1.imag();
				if (std::abs(c) >= std::abs(d)) {
					const F r = d / c;
					const F t = c + d * r;
					return R((p + q * r) / t, (q - p * r) / t);
				}
				else {
					const F r = c / d;
					const F t = d + c * r;
					return R((p * r + q) / t, (q * r - p) / t);
				}
			}
			else {
				return a / b;
			}
		}
		// xsimd's complex batch divide has the same overflow as the scalar path,
		// so keep SIMD for non-complex only — complex falls back to operator()
		// per element, which uses Smith's algorithm above.
		template <class B>
		auto simd_apply(const B& a, const B& b) const
			-> std::enable_if_t<!xtl::is_complex<typename B::value_type>::value, decltype(a / b)>
		{
			return a / b;
		}
	};

	// xtensor's abs_fun routes complex arrays through xsimd's batched abs, which
	// computes sqrt(re² + im²) and overflows when re² + im² > F's max even though
	// |z| fits (e.g. complex64 at |z| ≈ 1.8e19). The scalar fallback via
	// std::abs(complex) is hypot-safe; force the same behavior on the SIMD path.
	struct abs_fun {
		template <class T>
		constexpr auto operator()(const T& x) const {
			if constexpr (xtl::is_complex<T>::value) {
				return std::hypot(x.real(), x.imag());
			}
			else if constexpr (std::is_unsigned_v<T>) {
				return x;  // |unsigned| is identity
			}
			else {
				return std::abs(x);
			}
		}
	};

	// xt::sign on complex returns sign(re or im) + 0j (a "which axis" indicator),
	// but the array-api spec defines sign(z) = z / |z| for nonzero complex and 0
	// for zero. Override the complex branch only; non-complex falls through to
	// xtensor's existing impl. hypot avoids the re²+im² overflow that plain
	// |z| via sqrt would hit near the dtype's max magnitude.
	struct sign_fun {
		template <class T>
		constexpr auto operator()(T x) const {
			if constexpr (xtl::is_complex<T>::value) {
				using F = typename T::value_type;
				const F mag = std::hypot(x.real(), x.imag());
				if (mag == F(0)) return T(0, 0);
				return T(x.real() / mag, x.imag() / mag);
			}
			else {
				return xt::math::sign_impl<T>::run(x);
			}
		}
	};

	// std::log(complex<F>) computes 0.5 * log(re² + im²) + i·atan2(im, re),
	// so the real part overflows when re² + im² exceeds F's max even though
	// |z| is finite (e.g. complex64 with |z| ≈ 1.8e19, |z|² ≈ 3.4e38 — right
	// at float32 max). hypot scales internally, dodging the intermediate
	// overflow; this matches numpy's complex log.
	struct log_fun {
		template <class T>
		constexpr auto operator()(T arg) const {
			if constexpr (xtl::is_complex<T>::value) {
				using F = typename T::value_type;
				return T(std::log(std::hypot(arg.real(), arg.imag())), std::atan2(arg.imag(), arg.real()));
			}
			else {
				return std::log(arg);
			}
		}
	};

	// std::atan / std::atanh on complex use formulas that overflow or pick the
	// wrong branch at extreme magnitudes. Use the two-log identities — log(num)
	// and log(den) are each individually well-defined, while the ratio's
	// principal log is branch-cut-ambiguous when num/den ≈ -1. Non-complex
	// falls through to std::atan / std::atanh.
	struct atan_fun {
		template <class T>
		constexpr auto operator()(T z) const {
			if constexpr (xtl::is_complex<T>::value) {
				using F = typename T::value_type;
				// atan(z) = (1/(2i)) · (log(1 + iz) - log(1 - iz))
				const T iz(-z.imag(), z.real());
				const T num(F(1) + iz.real(), iz.imag());
				const T den(F(1) - iz.real(), -iz.imag());
				const T diff = log_fun{}(num) - log_fun{}(den);
				return T(diff.imag() / F(2), -diff.real() / F(2));
			}
			else {
				return std::atan(z);
			}
		}
	};

	struct atanh_fun {
		template <class T>
		constexpr auto operator()(T z) const {
			if constexpr (xtl::is_complex<T>::value) {
				using F = typename T::value_type;
				// atanh(z) = (1/2) · (log(1 + z) - log(1 - z))
				const T num(F(1) + z.real(), z.imag());
				const T den(F(1) - z.real(), -z.imag());
				const T diff = log_fun{}(num) - log_fun{}(den);
				return T(diff.real() / F(2), diff.imag() / F(2));
			}
			else {
				return std::atanh(z);
			}
		}
	};

	// std::log2 / std::log10 / std::log1p / std::expm1 don't take std::complex.
	// Compose via natural log / exp for complex; defer to the std specials for
	// real dtypes since they're more accurate near zero / for huge magnitudes
	// than the naive composition.
	struct log2_fun {
		template <class T>
		constexpr auto operator()(T arg) const {
			if constexpr (xtl::is_complex<T>::value) {
				using F = typename T::value_type;
				return std::log(arg) / static_cast<F>(0.6931471805599453);  // ln(2)
			}
			else {
				return std::log2(arg);
			}
		}
	};
	struct log10_fun {
		template <class T>
		constexpr auto operator()(T arg) const {
			if constexpr (xtl::is_complex<T>::value) {
				using F = typename T::value_type;
				return std::log(arg) / static_cast<F>(2.302585092994046);  // ln(10)
			}
			else {
				return std::log10(arg);
			}
		}
	};
	struct log1p_fun {
		template <class T>
		constexpr auto operator()(T arg) const {
			if constexpr (xtl::is_complex<T>::value) {
				return std::log(arg + T(1));
			}
			else {
				return std::log1p(arg);
			}
		}
	};
	struct expm1_fun {
		template <class T>
		constexpr auto operator()(T arg) const {
			if constexpr (xtl::is_complex<T>::value) {
				return std::exp(arg) - T(1);
			}
			else {
				return std::expm1(arg);
			}
		}
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
	XTENSOR_REDUCER_FUNCTION(va_any, xt::detail::logical_or, bool, false)
	XTENSOR_REDUCER_FUNCTION(va_all, xt::detail::logical_and, bool, true)

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
			// bool ← complex: nonzero in either component → true.
			return xt::cast<uint8_t>(xt::not_equal(cvalue, static_cast<VRead>(0)));
		}
		else if constexpr (xtl::is_complex<VRead>::value && !xtl::is_complex<VWrite>::value) {
			// real ← complex: drop the imaginary part. Numpy raises a
			// ComplexWarning here; emit the equivalent Godot warning so
			// users notice the silent truncation.
			WARN_PRINT_ONCE("Casting complex array to real dtype discards the imaginary part.");
			return xt::cast<VWrite>(xt::real(cvalue));
		}
		else if constexpr (xtl::is_complex<VWrite>::value) {
			// complex target — cover same-precision, cross-precision, and
			// real → complex in one branch via xt::cast (which xsimd also
			// requires for any → complex).
			return xt::cast<VWrite>(cvalue);
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
#endif
		else
		{
			return cvalue;
		}
	}

	template <class T1>
	auto consecutive(const void* start, const void* step, const std::size_t& num) {
		using V = typename T1::value_type;
		return *(const V*)start + xt::arange(num) * *(const V*)step;
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

#define IMPLEMENT_UNARY_AFUNC(UFUNC_NAME, FLAT, AXIS)\
template <typename R, typename A>\
inline void UFUNC_NAME(R& ret, const A& a, const va::axes_type* axes) {\
	if (axes) {\
		va::broadcasting_assign_typesafe(ret, AXIS);\
	}\
	else {\
		va::broadcasting_assign_typesafe(ret, FLAT);\
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
	IMPLEMENT_INPLACE_VFUNC(fill, xt::xscalar(*static_cast<typename R::value_type*>(fill_value)), void* fill_value)
	IMPLEMENT_UNARY_VFUNC(assign, va::op::vcast<typename R::value_type>(a))
	IMPLEMENT_INPLACE_VFUNC(fill_random_float, xt::random::rand<typename R::value_type>(ret.shape(), 0, 1, engine), xt::random::default_engine_type& engine)

	template<typename R>
	inline void fill_random_int(R& ret, xt::random::default_engine_type& engine, long long low, long long high) {
		using T = typename R::value_type;
		// TODO Should automatically figure out somehow which are supported, not hardcode it...
#ifdef _WIN32
		// Windows supports no 8 bit random
		using TRandom = std::conditional_t<
			std::is_same_v<T, int8_t>,
			int16_t,
			std::conditional_t<
				std::is_same_v<T, bool> || std::is_same_v<T, uint8_t>,
				uint16_t,
				T
			>
		>;
#else
		// Unix supports all integrals except bool
		using TRandom = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
#endif
		va::broadcasting_assign_typesafe(ret, xt::random::randint<TRandom>(ret.shape(), low, high, engine));
	}
	IMPLEMENT_INPLACE_VFUNC(fill_random_normal, xt::random::randn<typename R::value_type>(ret.shape(), 0, 1, engine), xt::random::default_engine_type& engine)

	IMPLEMENT_INPLACE_VFUNC(fill_consecutive, va::op::consecutive<R>(start, step, num), const void* start, const void* step, std::size_t num)

	IMPLEMENT_UNARY_VFUNC(negative, -va::promote::to_num(a))
	IMPLEMENT_UNARY_VFUNC(sign, xt::detail::make_xfunction<va::op::sign_fun>(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(abs, xt::detail::make_xfunction<va::op::abs_fun>(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(square, xt::square(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(sqrt, xt::sqrt(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(exp, xt::exp(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(expm1, xt::detail::make_xfunction<va::op::expm1_fun>(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(log, xt::detail::make_xfunction<va::op::log_fun>(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(log2, xt::detail::make_xfunction<va::op::log2_fun>(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(log10, xt::detail::make_xfunction<va::op::log10_fun>(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(log1p, xt::detail::make_xfunction<va::op::log1p_fun>(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(rad2deg, xt::rad2deg(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(deg2rad, xt::deg2rad(va::promote::to_num(a)))

	IMPLEMENT_UNARY_VFUNC(conjugate, xt::detail::make_xfunction<va::op::conj_fun>(va::promote::to_num(a)))

	// TODO calling xt::add directly etc. is broken because of the DEFINE_COMPLEX_OVERLOAD
	IMPLEMENT_BINARY_VFUNC(add, xt::detail::make_xfunction<xt::detail::plus>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(subtract, xt::detail::make_xfunction<xt::detail::minus>(va::promote::to_num(a), va::promote::to_num(b)));
	IMPLEMENT_BINARY_VFUNC(multiply, xt::detail::make_xfunction<xt::detail::multiplies>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(divide, xt::detail::make_xfunction<va::op::divide_fun>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(floor_divide, xt::detail::make_xfunction<va::op::floor_divide_fun>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(remainder, xt::remainder(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(pow, xt::pow(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(minimum, xt::minimum(a, b))
	IMPLEMENT_BINARY_VFUNC(maximum, xt::maximum(a, b))
	IMPLEMENT_BINARY_VFUNC(hypot, xt::hypot(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(copysign, xt::detail::make_xfunction<va::op::copysign_fun>(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_BINARY_VFUNC(logaddexp,
		xt::maximum(va::promote::to_num(a), va::promote::to_num(b))
			+ xt::log1p(xt::exp(-xt::abs(va::promote::to_num(a) - va::promote::to_num(b)))))
	IMPLEMENT_UNARY_VFUNC(signbit, xt::detail::make_xfunction<va::op::signbit_fun>(va::promote::to_num(a)))

	// sum/prod must accumulate at the output cell's dtype (the vfunc table maps
	// narrow ints to int64), or `prod(int32)` overflows during accumulation and
	// the wraparound propagates into the int64 result.
	IMPLEMENT_UNARY_RFUNC(sum, xt::sum<typename R::value_type>(a)(), xt::sum<typename R::value_type>(a, *axes))
	IMPLEMENT_UNARY_RFUNC(prod, xt::prod<typename R::value_type>(a)(), xt::prod<typename R::value_type>(a, *axes))
	IMPLEMENT_UNARY_AFUNC(cumsum, xt::cumsum<typename R::value_type>(a), xt::cumsum<typename R::value_type>(a, (*axes)[0]))
	IMPLEMENT_UNARY_AFUNC(cumprod, xt::cumprod<typename R::value_type>(a), xt::cumprod<typename R::value_type>(a, (*axes)[0]))
	IMPLEMENT_UNARY_VFUNC(diff, xt::diff(a, n, axis), std::size_t n, std::ptrdiff_t axis)
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
	IMPLEMENT_UNARY_VFUNC(atan, xt::detail::make_xfunction<va::op::atan_fun>(va::promote::to_num(a)))
	IMPLEMENT_BINARY_VFUNC(atan2, xt::atan2(va::promote::to_num(a), va::promote::to_num(b)))
	IMPLEMENT_UNARY_VFUNC(sinh, xt::sinh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(cosh, xt::cosh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(tanh, xt::tanh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(asinh, xt::asinh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(acosh, xt::acosh(va::promote::to_num(a)))
	IMPLEMENT_UNARY_VFUNC(atanh, xt::detail::make_xfunction<va::op::atanh_fun>(va::promote::to_num(a)))

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

	// Cast cond from bool to uint8_t (same memory layout) so the SIMD load uses
	// the uint8 path instead of the broken bool path; comparison to 0 then
	// produces a bool batch via the operation itself (cf. issue #123).
	IMPLEMENT_BINARY_VFUNC(where,
		xt::where(xt::not_equal(xt::cast<uint8_t>(*cond_ptr), static_cast<uint8_t>(0)), a, b),
		const compute_case<bool*>* cond_ptr)
	IMPLEMENT_BINARY_VFUNC(is_close, xt::isclose(va::promote::to_num(a), va::promote::to_num(b), rtol, atol, equal_nan), double rtol, double atol, bool equal_nan)
	IMPLEMENT_BINARY_VFUNC(array_equiv, xt::xscalar<bool>(xt::all(xt::equal(a, b))))
	IMPLEMENT_BINARY_VFUNC(all_close, xt::xscalar<bool>(xt::all(xt::isclose(va::promote::to_num(a), va::promote::to_num(b), rtol, atol, equal_nan))), double rtol, double atol, bool equal_nan)

	IMPLEMENT_UNARY_VFUNC(fft, xt::fft::fft(std::forward<decltype(a)>(a), axis), std::ptrdiff_t axis)
	IMPLEMENT_UNARY_VFUNC(
		pad,
		xt::pad(std::forward<decltype(a)>(a), pad_width, pad_mode, *static_cast<typename A::value_type*>(pad_value)),
		std::vector<std::vector<std::size_t>>& pad_width,
		xt::pad_mode pad_mode,
		void* pad_value
	)

	IMPLEMENT_BINARY_RFUNC(sum_product, xt::sum(a * b)(), xt::sum(a * b, *axes))

	IMPLEMENT_BINARY_VFUNC(a0xb1_minus_a1xb0,
		(xt::strided_view(a, { xt::ellipsis(), i0 }) * xt::strided_view(b, { xt::ellipsis(), i1 }))
		- (xt::strided_view(a, { xt::ellipsis(), i1 }) * xt::strided_view(b, { xt::ellipsis(), i0 })),
		const std::ptrdiff_t i0, const std::ptrdiff_t i1
	)
} // namespace va::vfunc::impl

#endif //VFUNCS_HPP
