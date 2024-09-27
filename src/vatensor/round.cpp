
#include "round.h"

#include <utility>                                       // for move
#include "vatensor/varray.h"                             // for VArray, VArr...
#include "vcompute.h"                            // for XFunction
#include "vpromote.h"                                    // for num_function...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xmath.hpp"                             // for ceil_fun
#include "xtensor/xoperation.hpp"                        // for make_xfunction

using namespace va;

void va::ceil(const VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::ceil_fun>>(
        va::XFunction<xt::math::ceil_fun> {},
        target,
        array.compute_read()
    );
#endif
}

void va::floor(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::floor_fun>>(
        va::XFunction<xt::math::floor_fun> {},
        target,
        array.compute_read()
    );
#endif
}

void va::trunc(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::trunc_fun>>(
        va::XFunction<xt::math::trunc_fun> {},
        target,
        array.compute_read()
    );
#endif
}

void va::round(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::round_fun>>(
        va::XFunction<xt::math::round_fun> {},
        target,
        array.compute_read()
    );
#endif
}

void va::nearbyint(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_ROUNDING_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ROUNDING_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::nearbyint_fun>>(
        va::XFunction<xt::math::nearbyint_fun> {},
        target,
        array.compute_read()
    );
#endif
}
