#include "comparison.hpp"

#include <utility>                                       // for move
#include "scalar_tricks.hpp"
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for common_num_i...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xoperation.hpp"                        // for equal_to

using namespace va;

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
#if !(defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(_M_ARM))
// FIXME NEON xtensor / xsimd has a compile-time bug, see
// https://github.com/xtensor-stack/xtensor/issues/2733
void equal_to(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
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
#endif
#endif

void va::equal_to(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION

// Doesn't work right now with NEON, see above.
#if !(defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(_M_ARM))
	OPTIMIZE_COMMUTATIVE(::equal_to, allocator, target, a, b);
#endif

#endif

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

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
#if !(defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(_M_ARM))
void not_equal_to(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
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
#endif
#endif

void va::not_equal_to(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
#if !(defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(_M_ARM))
	OPTIMIZE_COMMUTATIVE(::not_equal_to, allocator, target, a, b);
#endif
#endif

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

void va::greater(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		va::xoperation_inplace<
			Feature::greater,
			promote::reject_complex<promote::num_in_nat_out>
		>(
			va::XFunction<xt::detail::greater> {},
			allocator,
			target,
			va::to_single_value(a),
			b
		);
		return;
	}
	if (va::dimension(b) == 0) {
		va::xoperation_inplace<
			Feature::greater,
			promote::reject_complex<promote::num_in_nat_out>
		>(
			va::XFunction<xt::detail::greater> {},
			allocator,
			target,
			a,
			va::to_single_value(b)
		);
		return;
	}
#endif

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

void va::greater_equal(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		va::xoperation_inplace<
			Feature::greater_equal,
			promote::reject_complex<promote::num_in_nat_out>
		>(
			va::XFunction<xt::detail::greater_equal> {},
			allocator,
			target,
			va::to_single_value(a),
			b
		);
		return;
	}
	if (va::dimension(b) == 0) {
		va::xoperation_inplace<
			Feature::greater_equal,
			promote::reject_complex<promote::num_in_nat_out>
		>(
			va::XFunction<xt::detail::greater_equal> {},
			allocator,
			target,
			a,
			va::to_single_value(b)
		);
		return;
	}
#endif

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

void va::less(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		va::xoperation_inplace<
			Feature::less,
			promote::reject_complex<promote::num_in_nat_out>
		>(
			va::XFunction<xt::detail::less> {},
			allocator,
			target,
			va::to_single_value(a),
			b
		);
		return;
	}
	if (va::dimension(b) == 0) {
		va::xoperation_inplace<
			Feature::less,
			promote::reject_complex<promote::num_in_nat_out>
		>(
			va::XFunction<xt::detail::less> {},
			allocator,
			target,
			a,
			va::to_single_value(b)
		);
		return;
	}
#endif

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

void va::less_equal(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		va::xoperation_inplace<
			Feature::less_equal,
			promote::reject_complex<promote::num_in_nat_out>
		>(
			va::XFunction<xt::detail::less_equal> {},
			allocator,
			target,
			va::to_single_value(a),
			b
		);
		return;
	}
	if (va::dimension(b) == 0) {
		va::xoperation_inplace<
			Feature::less_equal,
			promote::reject_complex<promote::num_in_nat_out>
		>(
			va::XFunction<xt::detail::less_equal> {},
			allocator,
			target,
			a,
			va::to_single_value(b)
		);
		return;
	}
#endif

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
