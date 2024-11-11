#include "comparison.hpp"

#include <utility>                                       // for move
#include "scalar_tricks.hpp"
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for common_num_i...
#include "xtensor/xmath.hpp"                           // for layout_type
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xoperation.hpp"                        // for equal_to

using namespace va;

template <typename A, typename B>
void equal_to(VStoreAllocator& allocator, VArrayTarget target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::equal_to,
		promote::common_in_nat_out
	>(
		va::XFunction<xt::detail::equal_to> {},
		allocator,
		target,
		a,
		b
	);
}

void va::equal_to(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION

// FIXME NEON xtensor / xsimd has a compile-time bug, see
// https://github.com/xtensor-stack/xtensor/issues/2733
#if !(defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(_M_ARM))
	OPTIMIZE_COMMUTATIVE(::equal_to, allocator, target, a, b);
#endif

#endif

	::equal_to(allocator, target, a, b);
}

template <typename A, typename B>
void not_equal_to(VStoreAllocator& allocator, VArrayTarget target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::not_equal_to,
		promote::common_in_nat_out
	>(
		va::XFunction<xt::detail::not_equal_to> {},
		allocator,
		target,
		a,
		b
	);
}

void va::not_equal_to(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
#if !(defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(_M_ARM))
	OPTIMIZE_COMMUTATIVE(::not_equal_to, allocator, target, a, b);
#endif
#endif

	::not_equal_to(allocator, target, a, b);
}

template <typename A, typename B>
void greater(VStoreAllocator& allocator, VArrayTarget target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::greater,
		promote::reject_complex<promote::num_in_nat_out>
	>(
		va::XFunction<xt::detail::greater> {},
		allocator,
		target,
		a,
		b
	);
}

void va::greater(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_NONCOMMUTATIVE(::greater, allocator, target, a, b);
#endif

	::greater(allocator, target, a, b);
}

template <typename A, typename B>
void greater_equal(VStoreAllocator& allocator, VArrayTarget target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::greater_equal,
		promote::reject_complex<promote::num_in_nat_out>
	>(
		va::XFunction<xt::detail::greater_equal> {},
		allocator,
		target,
		a,
		b
	);
}

void va::greater_equal(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_NONCOMMUTATIVE(::greater_equal, allocator, target, a, b);
#endif

	::greater_equal(allocator, target, a, b);
}

template <typename A, typename B>
void less(VStoreAllocator& allocator, VArrayTarget target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::less,
		promote::reject_complex<promote::num_in_nat_out>
	>(
		va::XFunction<xt::detail::less> {},
		allocator,
		target,
		a,
		b
	);
}

void va::less(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_NONCOMMUTATIVE(::less, allocator, target, a, b);
#endif

	::less(allocator, target, a, b);
}

template <typename A, typename B>
void less_equal(VStoreAllocator& allocator, VArrayTarget target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::less_equal,
		promote::reject_complex<promote::num_in_nat_out>
	>(
		va::XFunction<xt::detail::less_equal> {},
		allocator,
		target,
		a,
		b
	);
}

void va::less_equal(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_NONCOMMUTATIVE(::less_equal, allocator, target, a, b);
#endif

	::less_equal(allocator, target, a, b);
}

template <typename A, typename B>
void is_close(VStoreAllocator& allocator, VArrayTarget target, const A& a, const B& b, double rtol, double atol, bool equal_nan) {
	va::xoperation_inplace<
		Feature::is_close,
		promote::common_in_nat_out
	>(
		[rtol, atol, equal_nan](auto&& a, auto&& b) {
			return xt::isclose(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b), rtol, atol, equal_nan);
		},
		allocator,
		target,
		a,
		b
	);
}

void va::is_close(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b, double rtol, double atol, bool equal_nan) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		::is_close(allocator, target, b, va::to_single_value(a), rtol, atol, equal_nan);
		return;
	}
	if (va::dimension(b) == 0) {
		::is_close(allocator, target, a, va::to_single_value(b), rtol, atol, equal_nan);
		return;
	}
#endif

	::is_close(allocator, target, a, b, rtol, atol, equal_nan);
}

void va::is_nan(VStoreAllocator& allocator, VArrayTarget target, const VData& a) {
	va::xoperation_inplace<
		Feature::is_nan,
		promote::num_in_nat_out
	>(
		va::XFunction<xt::math::isnan_fun> {},
		allocator,
		target,
		a
	);
}

void va::is_finite(VStoreAllocator& allocator, VArrayTarget target, const VData& a) {
	va::xoperation_inplace<
		Feature::is_finite,
		promote::num_in_nat_out
	>(
		va::XFunction<xt::math::isfinite_fun> {},
		allocator,
		target,
		a
	);
}

void va::is_inf(VStoreAllocator& allocator, VArrayTarget target, const VData& a) {
	va::xoperation_inplace<
		Feature::is_inf,
		promote::num_in_nat_out
	>(
		va::XFunction<xt::math::isinf_fun> {},
		allocator,
		target,
		a
	);
}

bool va::array_equal(const VData& a, const VData& b) {
	return va::vreduce<
		Feature::array_equal,
		promote::common_in_nat_out,
		bool
	>(
		[](auto&& a, auto&& b) -> bool {
			return xt::all(xt::equal(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)));
		},
		a,
		b
	);
}

template <typename A, typename B>
bool all_close(const A& a, const B& b, double rtol, double atol, bool equal_nan) {
	return va::vreduce<
		Feature::all_close,
		promote::common_in_nat_out,
		bool
	>(
		[rtol, atol, equal_nan](auto&& a, auto&& b) -> bool {
			return xt::all(xt::isclose(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b), rtol, atol, equal_nan));
		},
		a,
		b
	);
}

bool va::all_close(const VData& a, const VData& b, double rtol, double atol, bool equal_nan) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		return ::all_close(b, va::to_single_value(a), rtol, atol, equal_nan);
	}
	if (va::dimension(b) == 0) {
		return ::all_close(a, va::to_single_value(b), rtol, atol, equal_nan);
	}
#endif

	return ::all_close(a, b, rtol, atol, equal_nan);
}
