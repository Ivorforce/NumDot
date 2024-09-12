
#include "round.h"

#include "vatensor/varray.h"                   
#include "vcompute.h"                                   
// #include "xtensor/xlayout.hpp"                           
// #include "xtensor/xmath.hpp"                             
// #include "xtensor/xoperation.hpp"                       

using namespace va;

VArray va::ceil(const VArray &array) {
    return xoperation<promote::function_result<xt::math::ceil_fun>>(va::XFunction<xt::math::ceil_fun> {}, array.to_compute_variant());
}

VArray va::floor(const VArray &array) {
    return xoperation<promote::function_result<xt::math::floor_fun>>(va::XFunction<xt::math::floor_fun> {}, array.to_compute_variant());
}

VArray va::trunc(const VArray &array) {
    return xoperation<promote::function_result<xt::math::trunc_fun>>(va::XFunction<xt::math::trunc_fun> {}, array.to_compute_variant());
}

VArray va::round(const VArray &array) {
    return xoperation<promote::function_result<xt::math::round_fun>>(va::XFunction<xt::math::round_fun> {}, array.to_compute_variant());
}

VArray va::nearbyint(const VArray &array) {
    return xoperation<promote::function_result<xt::math::nearbyint_fun>>(va::XFunction<xt::math::nearbyint_fun> {}, array.to_compute_variant());
}

VArray va::rint(const VArray &array) {
    return xoperation<promote::function_result<xt::math::rint_fun>>(va::XFunction<xt::math::rint_fun> {}, array.to_compute_variant());
}
