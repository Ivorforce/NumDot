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
void equal_to(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::common_in_bool_out>(
		va::XFunction<xt::detail::equal_to> {},
		target,
		a.read,
		b
	);
}
#endif

void va::equal_to(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_COMPARISON_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_COMPARISON_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	// FIXME NEON xtensor / xsimd has a compile-time bug, see
	// https://github.com/xtensor-stack/xtensor/issues/2733
#ifndef __ARM_NEON__
	OPTIMIZE_COMMUTATIVE(::equal_to, a, b);
#endif

#endif

	va::xoperation_inplace<promote::common_in_bool_out>(
		va::XFunction<xt::detail::equal_to> {},
		target,
		a.read,
		b.read
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void not_equal_to(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::common_in_bool_out>(
		va::XFunction<xt::detail::not_equal_to> {},
		target,
		a.read,
		b
	);
}
#endif

void va::not_equal_to(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_COMPARISON_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_COMPARISON_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::not_equal_to, a, b);
#endif

	va::xoperation_inplace<promote::common_in_bool_out>(
		va::XFunction<xt::detail::not_equal_to> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::greater(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_COMPARISON_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_COMPARISON_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
			va::XFunction<xt::detail::greater> {},
			target,
			a.to_single_value(),
			b.read
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
			va::XFunction<xt::detail::greater> {},
			target,
			a.read,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
		va::XFunction<xt::detail::greater> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::greater_equal(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_COMPARISON_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_COMPARISON_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
			va::XFunction<xt::detail::greater_equal> {},
			target,
			a.to_single_value(),
			b.read
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
			va::XFunction<xt::detail::greater_equal> {},
			target,
			a.read,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
		va::XFunction<xt::detail::greater_equal> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::less(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_COMPARISON_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_COMPARISON_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
			va::XFunction<xt::detail::less> {},
			target,
			a.to_single_value(),
			b.read
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
			va::XFunction<xt::detail::less> {},
			target,
			a.read,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
		va::XFunction<xt::detail::less> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::less_equal(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_COMPARISON_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_COMPARISON_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
			va::XFunction<xt::detail::less_equal> {},
			target,
			a.to_single_value(),
			b.read
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
			va::XFunction<xt::detail::less_equal> {},
			target,
			a.read,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
		va::XFunction<xt::detail::less_equal> {},
		target,
		a.read,
		b.read
	);
#endif
}
