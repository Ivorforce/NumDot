#include "logical.hpp"

#include <utility>                                      // for move
#include "scalar_tricks.hpp"
#include "varray.hpp"                            // for VArray, VArra...
#include "vcompute.hpp"                           // for XFunction
#include "vpromote.hpp"                                   // for bool_in_bool_out
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xoperation.hpp"                       // for logical_and

using namespace va;

void assign_bool(VStoreAllocator& allocator, VArrayTarget target, const VArray& a) {
	if (a.dtype() == Bool) {
		va::assign(allocator, target, a.data);
		return;
	}

	va::xoperation_inplace<promote::x_in_nat_out<bool>>(
		XFunction<typename xt::detail::cast<bool>::functor> {},
		allocator,
		target,
		a.data
	);
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_and(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	// Can shortcut the logic
	if (!scalar_to_type<bool>(b)) {
		assign(target, false);
		return;
	}

	assign_bool(allocator, target, a);
}
#endif

void va::logical_and(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_LOGICAL_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_LOGICAL_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_and, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::reject_complex<promote::x_in_nat_out<bool>>>(
		XFunction<xt::detail::logical_and> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_or(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	// Can shortcut the logic
	if (scalar_to_type<bool>(b)) {
		assign(target, true);
		return;
	}

	assign_bool(allocator, target, a);
}
#endif

void va::logical_or(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_LOGICAL_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_LOGICAL_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_or, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::reject_complex<promote::x_in_nat_out<bool>>>(
		XFunction<xt::detail::logical_or> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_xor(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	// Can shortcut the logic
	if (scalar_to_type<bool>(b)) {
		va::logical_not(allocator, target, a);
		return;
	}

	assign_bool(allocator, target, a);
}
#endif

void va::logical_xor(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_LOGICAL_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_LOGICAL_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_xor, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::reject_complex<promote::x_in_nat_out<bool>>>(
		XFunction<xt::detail::not_equal_to> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void va::logical_not(VStoreAllocator& allocator, VArrayTarget target, const VArray& a) {
#ifdef NUMDOT_DISABLE_LOGICAL_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_LOGICAL_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::reject_complex<promote::x_in_nat_out<bool>>>(
		XFunction<xt::detail::logical_not> {},
		allocator,
		target,
		a.data
	);
#endif
}
