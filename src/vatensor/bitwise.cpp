#include "bitwise.hpp"

#include <utility>                                      // for move
#include "scalar_tricks.hpp"
#include "varray.hpp"                            // for VArray, VArra...
#include "vcompute.hpp"                           // for XFunction
#include "vpromote.hpp"                                   // for bool_in_bool_out
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xoperation.hpp"                       // for logical_and

using namespace va;

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void bitwise_and(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::common_int_in_same_out>(
		va::XFunction<xt::detail::bitwise_and> {},
		allocator,
		target,
		a.data,
		b
	);
}
#endif

void va::bitwise_and(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_BITWISE_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_BITWISE_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::bitwise_and, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::common_int_in_same_out>(
		XFunction<xt::detail::bitwise_and> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void bitwise_or(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::common_int_in_same_out>(
		va::XFunction<xt::detail::bitwise_or> {},
		allocator,
		target,
		a.data,
		b
	);
}
#endif

void va::bitwise_or(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_BITWISE_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_BITWISE_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::bitwise_or, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::common_int_in_same_out>(
		XFunction<xt::detail::bitwise_or> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void bitwise_xor(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::common_int_in_same_out>(
		va::XFunction<xt::detail::bitwise_xor> {},
		allocator,
		target,
		a.data,
		b
	);
}
#endif

void va::bitwise_xor(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_BITWISE_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_BITWISE_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::bitwise_xor, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::common_int_in_same_out>(
		XFunction<xt::detail::not_equal_to> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void va::bitwise_not(VStoreAllocator& allocator, VArrayTarget target, const VArray& a) {
#ifdef NUMDOT_DISABLE_BITWISE_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_BITWISE_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::common_int_in_same_out>(
		XFunction<xt::detail::bitwise_not> {},
		allocator,
		target,
		a.data
	);
#endif
}
