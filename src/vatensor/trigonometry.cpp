#include "trigonometry.h"

#include "vatensor/varray.h"                             // for VArray
#include "vcompute.h"                                    // for XFunction
#include "vpromote.h"                                    // for promote
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xmath.hpp"                             // for cos_fun, sin...
#include "xtensor/xoperation.hpp"                        // for make_xfunction

using namespace va;

VArray va::sin(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::sin_fun>>(va::XFunction<xt::math::sin_fun> {}, array.to_compute_variant());
}

VArray va::cos(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::cos_fun>>(va::XFunction<xt::math::cos_fun> {}, array.to_compute_variant());
}

VArray va::tan(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::tan_fun>>(va::XFunction<xt::math::tan_fun> {}, array.to_compute_variant());
}

VArray va::asin(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::asin_fun>>(va::XFunction<xt::math::asin_fun> {}, array.to_compute_variant());
}

VArray va::acos(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::acos_fun>>(va::XFunction<xt::math::acos_fun> {}, array.to_compute_variant());
}

VArray va::atan(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::atan_fun>>(va::XFunction<xt::math::atan_fun> {}, array.to_compute_variant());
}

VArray va::atan2(const VArray &x1, const VArray &x2) {
    return xoperation<promote::num_function_result<xt::math::atan2_fun>>(va::XFunction<xt::math::atan2_fun> {}, x1.to_compute_variant(), x2.to_compute_variant());
}

VArray va::sinh(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::sinh_fun>>(va::XFunction<xt::math::sinh_fun> {}, array.to_compute_variant());
}

VArray va::cosh(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::cosh_fun>>(va::XFunction<xt::math::cosh_fun> {}, array.to_compute_variant());
}

VArray va::tanh(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::tanh_fun>>(va::XFunction<xt::math::tanh_fun> {}, array.to_compute_variant());
}

VArray va::asinh(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::asinh_fun>>(va::XFunction<xt::math::asinh_fun> {}, array.to_compute_variant());
}

VArray va::acosh(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::acosh_fun>>(va::XFunction<xt::math::acosh_fun> {}, array.to_compute_variant());
}

VArray va::atanh(const VArray &array) {
    return xoperation<promote::num_function_result<xt::math::atanh_fun>>(va::XFunction<xt::math::atanh_fun> {}, array.to_compute_variant());
}
