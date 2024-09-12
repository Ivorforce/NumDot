#include "trigonometry.h"

#include "vcompute.h"

using namespace va;

VArray va::sin(const VArray &array) {
    return xoperation<promote::common_type>(va::XFunction<xt::math::sin_fun> {}, array.to_compute_variant());
}

VArray va::cos(const VArray &array) {
    return xoperation<promote::common_type>(va::XFunction<xt::math::cos_fun> {}, array.to_compute_variant());
}

VArray va::tan(const VArray &array) {
    return xoperation<promote::common_type>(va::XFunction<xt::math::tan_fun> {}, array.to_compute_variant());
}
