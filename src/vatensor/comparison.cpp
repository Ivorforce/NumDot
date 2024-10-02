#include "comparison.hpp"

#include <utility>                                       // for move
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for common_num_i...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xoperation.hpp"                        // for equal_to

using namespace va;

void va::equal_to(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_COMPARISON_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_COMPARISON_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::common_in_bool_out>(
		va::XFunction<xt::detail::equal_to> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::not_equal_to(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_COMPARISON_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_COMPARISON_FUNCTIONS to enable it.");
#else
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
	va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
		va::XFunction<xt::detail::less_equal> {},
		target,
		a.read,
		b.read
	);
#endif
}
