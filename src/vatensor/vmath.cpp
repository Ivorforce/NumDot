#include "vmath.h"

#include "vatensor/varray.h"                                // for VArray
#include "vcompute.h"                                       // for XFunction
#include "vpromote.h"                                    // for promote
#include "xtensor/xlayout.hpp"                              // for layout_type
#include "xtensor/xmath.hpp"                                // for pow_fun
#include "xtensor/xoperation.hpp"                           // for divides

using namespace va;

VArray va::add(const VArray &a, const VArray &b) {
    return va::xoperation<promote::num_function_result<xt::detail::plus>>(
        XFunction<xt::detail::plus> {},
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

VArray va::subtract(const VArray &a, const VArray &b) {
    return va::xoperation<promote::num_function_result<xt::detail::minus>>(
        XFunction<xt::detail::minus> {},
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

VArray va::multiply(const VArray &a, const VArray &b) {
    return va::xoperation<promote::num_function_result<xt::detail::multiplies>>(
        XFunction<xt::detail::multiplies> {},
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

VArray va::divide(const VArray &a, const VArray &b) {
    return va::xoperation<promote::num_function_result<xt::detail::divides>>(
        XFunction<xt::detail::divides> {},
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

VArray va::remainder(const VArray &a, const VArray &b) {
    return va::xoperation<promote::num_function_result<xt::math::remainder_fun>>(
        XFunction<xt::math::remainder_fun> {},
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

VArray va::pow(const VArray &a, const VArray &b) {
    return va::xoperation<promote::num_function_result<xt::math::pow_fun>>(
        XFunction<xt::math::pow_fun> {},
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

VArray va::sign(const VArray &array) {
    return xoperation<promote::num_common_type>(va::XFunction<xt::math::sign_fun> {}, array.to_compute_variant());
}

VArray va::abs(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::abs_fun>>(va::XFunction<xt::math::abs_fun> {}, array.to_compute_variant());
}

VArray va::sqrt(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::sqrt_fun>>(va::XFunction<xt::math::sqrt_fun> {}, array.to_compute_variant());
}

VArray va::exp(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::exp_fun>>(va::XFunction<xt::math::exp_fun> {}, array.to_compute_variant());
}

VArray va::log(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::log_fun>>(va::XFunction<xt::math::log_fun> {}, array.to_compute_variant());
}
