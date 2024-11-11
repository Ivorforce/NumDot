#include "logical.hpp"

#include <utility>                                      // for move

#include "create.hpp"
#include "scalar_tricks.hpp"
#include "varray.hpp"                            // for VArray, VArra...
#include "vcompute.hpp"                           // for XFunction
#include "vpromote.hpp"                                   // for bool_in_bool_out
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xoperation.hpp"                       // for logical_and

using namespace va;

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_and(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
	// Can shortcut the logic
	if (!static_cast_scalar<bool>(b)) {
		assign(target, false);
		return;
	}

	va::assign_cast(allocator, target, a, DType::Bool);
}
#endif

void va::logical_and(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_and, allocator, target, a, b);
#endif

	va::xoperation_inplace<
		Feature::logical_and,
		promote::reject_complex<promote::x_in_nat_out<bool>>
	>(
		XFunction<xt::detail::logical_and> {},
		allocator,
		target,
		a,
		b
	);
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_or(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
	// Can shortcut the logic
	if (static_cast_scalar<bool>(b)) {
		assign(target, true);
		return;
	}

	va::assign_cast(allocator, target, a, DType::Bool);
}
#endif

void va::logical_or(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_or, allocator, target, a, b);
#endif

	va::xoperation_inplace<
		Feature::logical_or,
		promote::reject_complex<promote::x_in_nat_out<bool>>
	>(
		XFunction<xt::detail::logical_or> {},
		allocator,
		target,
		a,
		b
	);
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_xor(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
	// Can shortcut the logic
	if (static_cast_scalar<bool>(b)) {
		va::logical_not(allocator, target, a);
		return;
	}

	va::assign_cast(allocator, target, a, DType::Bool);
}
#endif

void va::logical_xor(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_xor, allocator, target, a, b);
#endif

	va::xoperation_inplace<
		Feature::logical_xor,
		promote::reject_complex<promote::x_in_nat_out<bool>>
	>(
		XFunction<xt::detail::not_equal_to> {},
		allocator,
		target,
		a,
		b
	);
}

void va::logical_not(VStoreAllocator& allocator, VArrayTarget target, const VData& a) {
	va::xoperation_inplace<
		Feature::logical_not,
		promote::reject_complex<promote::x_in_nat_out<bool>>
	>(
		XFunction<xt::detail::logical_not> {},
		allocator,
		target,
		a
	);
}
