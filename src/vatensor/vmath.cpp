#include "vmath.h"

#include <utility>                                          // for move
#include "vatensor/varray.h"                                // for VArray
#include "vcompute.h"                               // for XFunction
#include "vpromote.h"                                       // for num_funct...
#include "xtensor/xlayout.hpp"                              // for layout_type
#include "xtensor/xmath.hpp"                                // for maximum
#include "xtensor/xoperation.hpp"                           // for divides
#include "xtl/xfunctional.hpp"                              // for select

using namespace va;

void va::add(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::num_function_result<xt::detail::plus>>(
        XFunction<xt::detail::plus> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
#endif
}

void va::subtract(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::num_function_result<xt::detail::minus>>(
        XFunction<xt::detail::minus> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
#endif
}

void va::multiply(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::num_function_result<xt::detail::multiplies>>(
        XFunction<xt::detail::multiplies> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
#endif
}

void va::divide(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::num_function_result<xt::detail::divides>>(
        XFunction<xt::detail::divides> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
#endif
}

void va::remainder(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::num_function_result<xt::math::remainder_fun>>(
        XFunction<xt::math::remainder_fun> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
#endif
}

void va::pow(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::num_function_result<xt::math::pow_fun>>(
        XFunction<xt::math::pow_fun> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
#endif
}

void va::minimum(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::common_in_common_out>(
        XFunction<xt::math::minimum<void>> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
#endif
}

void va::maximum(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::common_in_common_out>(
        XFunction<xt::math::maximum<void>> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
#endif
}

void va::clip(VArrayTarget target, const VArray& a, const VArray& lo, const VArray& hi) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    va::xoperation_inplace<promote::common_in_common_out>(
        XFunction<xt::math::clamp_fun> {},
        target,
        a.to_compute_variant(),
        lo.to_compute_variant(),
        hi.to_compute_variant()
    );
#endif
}

void va::sign(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_common_type>(
        va::XFunction<xt::math::sign_fun> {},
        target,
        array.to_compute_variant()
    );
#endif
}

void va::abs(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::abs_fun>>(
        va::XFunction<xt::math::abs_fun> {},
        target,
        array.to_compute_variant()
    );
#endif
}

// TODO xt::square uses xt::square_fct, which is hidden behind an ifdef.
//  This function is rather harmless, so no idea why, but we can just re-declare it.
struct square_fun {
    template <class T>
    auto operator()(T x) const -> decltype(x * x) {
        return x * x;
    }
};

void va::square(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_common_type>(
        va::XFunction<square_fun> {},
        target,
        array.to_compute_variant()
    );
#endif
}

void va::sqrt(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::sqrt_fun>>(
        va::XFunction<xt::math::sqrt_fun> {},
        target,
        array.to_compute_variant()
    );
#endif
}

void va::exp(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::exp_fun>>(
        va::XFunction<xt::math::exp_fun> {},
        target,
        array.to_compute_variant()
    );
#endif
}

void va::log(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::log_fun>>(
        va::XFunction<xt::math::log_fun> {},
        target,
        array.to_compute_variant()
    );
#endif
}

void va::rad2deg(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::rad2deg>>(
        va::XFunction<xt::math::rad2deg> {},
        target,
        array.to_compute_variant()
    );
#endif
}

void va::deg2rad(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
    xoperation_inplace<promote::num_function_result<xt::math::deg2rad>>(
        va::XFunction<xt::math::deg2rad> {},
        target,
        array.to_compute_variant()
    );
#endif
}