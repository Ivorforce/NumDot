#include "round.hpp"

#include <utility>                                       // for move
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for num_function...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xmath.hpp"                             // for ceil_fun
#include "xtensor/xoperation.hpp"                        // for make_xfunction

using namespace va;

void va::ceil(const VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::ceil_fun>>>(
		va::XFunction<xt::math::ceil_fun> {},
		target,
		array.data
	);
#endif
}

void va::floor(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::floor_fun>>>(
		va::XFunction<xt::math::floor_fun> {},
		target,
		array.data
	);
#endif
}

void va::trunc(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::trunc_fun>>>(
		va::XFunction<xt::math::trunc_fun> {},
		target,
		array.data
	);
#endif
}

void va::round(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::round_fun>>>(
		va::XFunction<xt::math::round_fun> {},
		target,
		array.data
	);
#endif
}

void va::nearbyint(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::nearbyint_fun>>>(
		va::XFunction<xt::math::nearbyint_fun> {},
		target,
		array.data
	);
#endif
}
