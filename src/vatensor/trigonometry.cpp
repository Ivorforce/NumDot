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
