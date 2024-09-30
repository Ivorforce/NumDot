#include "trigonometry.h"

#include <utility>                                       // for move
#include "vatensor/varray.h"                             // for VArray, VArr...
#include "vcompute.h"                            // for XFunction
#include "vpromote.h"                                    // for num_function...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xmath.hpp"                             // for atan2_fun
#include "xtensor/xoperation.hpp"                        // for make_xfunction

using namespace va;

void va::sin(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::sin_fun>>(
        va::XFunction<xt::math::sin_fun> {},
        target,
        array.read
    );
#endif
}

void va::cos(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::cos_fun>>(
        va::XFunction<xt::math::cos_fun> {},
        target,
        array.read
    );
#endif
}

void va::tan(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::tan_fun>>(
        va::XFunction<xt::math::tan_fun> {},
        target,
        array.read
    );
#endif
}

void va::asin(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::asin_fun>>(
        va::XFunction<xt::math::asin_fun> {},
        target,
        array.read
    );
#endif
}

void va::acos(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::acos_fun>>(
        va::XFunction<xt::math::acos_fun> {},
        target,
        array.read
    );
#endif
}

void va::atan(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::atan_fun>>(
        va::XFunction<xt::math::atan_fun> {},
        target,
        array.read
    );
#endif
}

void va::atan2(VArrayTarget target, const VArray& x1, const VArray& x2) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::atan2_fun>>(
        va::XFunction<xt::math::atan2_fun> {},
        target,
        x1.read,
        x2.read
    );
#endif
}

void va::sinh(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::sinh_fun>>(
        va::XFunction<xt::math::sinh_fun> {},
        target,
        array.read
    );
#endif
}

void va::cosh(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::cosh_fun>>(
        va::XFunction<xt::math::cosh_fun> {},
        target,
        array.read
    );
#endif
}

void va::tanh(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::tanh_fun>>(
        va::XFunction<xt::math::tanh_fun> {},
        target,
        array.read
    );
#endif
}

void va::asinh(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::asinh_fun>>(
        va::XFunction<xt::math::asinh_fun> {},
        target,
        array.read
    );
#endif
}

void va::acosh(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::acosh_fun>>(
        va::XFunction<xt::math::acosh_fun> {},
        target,
        array.read
    );
#endif
}

void va::atanh(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::atanh_fun>>(
        va::XFunction<xt::math::atanh_fun> {},
        target,
        array.read
    );
#endif
}
