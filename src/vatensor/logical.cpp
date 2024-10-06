#include "logical.hpp"

#include <utility>                                      // for move
#include "scalar_tricks.hpp"
#include "varray.hpp"                            // for VArray, VArra...
#include "vcompute.hpp"                           // for XFunction
#include "vpromote.hpp"                                   // for bool_in_bool_out
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xoperation.hpp"                       // for logical_and

using namespace va;

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_and(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::bool_in_bool_out>(
		XFunction<xt::detail::logical_and> {},
		target,
		a.read,
		b
	);
}
#endif

void va::logical_and(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_LOGICAL_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_LOGICAL_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_and, a, b);
#endif

	va::xoperation_inplace<promote::bool_in_bool_out>(
		XFunction<xt::detail::logical_and> {},
		target,
		a.read,
		b.read
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_or(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::bool_in_bool_out>(
		XFunction<xt::detail::logical_or> {},
		target,
		a.read,
		b
	);
}
#endif

void va::logical_or(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_LOGICAL_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_LOGICAL_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_or, a, b);
#endif

	va::xoperation_inplace<promote::bool_in_bool_out>(
		XFunction<xt::detail::logical_or> {},
		target,
		a.read,
		b.read
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void logical_xor(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::bool_in_bool_out>(
		XFunction<xt::detail::not_equal_to> {},
		target,
		a.read,
		b
	);
}
#endif

void va::logical_xor(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_LOGICAL_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_LOGICAL_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::logical_xor, a, b);
#endif

	va::xoperation_inplace<promote::bool_in_bool_out>(
		XFunction<xt::detail::not_equal_to> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::logical_not(VArrayTarget target, const VArray& a) {
#ifdef NUMDOT_DISABLE_LOGICAL_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_LOGICAL_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::bool_in_bool_out>(
		XFunction<xt::detail::logical_not> {},
		target,
		a.read
	);
#endif
}
