#include "comparison.h"

#include "vatensor/varray.h"                            // for VArray
#include "vcompute.h"                                   // for XFunction
#include "vpromote.h"                                   // for identity_in_b...
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xoperation.hpp"                       // for equal_to, mak...

using namespace va;

VArray va::equal_to(const VArray &a, const VArray &b) {
    return xoperation<promote::common_in_bool_out>(va::XFunction<xt::detail::equal_to> {}, a.to_compute_variant(), b.to_compute_variant());
}

VArray va::not_equal_to(const VArray &a, const VArray &b) {
    return xoperation<promote::common_in_bool_out>(va::XFunction<xt::detail::not_equal_to> {}, a.to_compute_variant(), b.to_compute_variant());
}

VArray va::greater(const VArray &a, const VArray &b) {
    return xoperation<promote::common_num_in_x_out<bool>>(va::XFunction<xt::detail::greater> {}, a.to_compute_variant(), b.to_compute_variant());
}

VArray va::greater_equal(const VArray &a, const VArray &b) {
    return xoperation<promote::common_num_in_x_out<bool>>(va::XFunction<xt::detail::greater_equal> {}, a.to_compute_variant(), b.to_compute_variant());
}

VArray va::less(const VArray &a, const VArray &b) {
    return xoperation<promote::common_num_in_x_out<bool>>(va::XFunction<xt::detail::less> {}, a.to_compute_variant(), b.to_compute_variant());
}

VArray va::less_equal(const VArray &a, const VArray &b) {
    return xoperation<promote::common_num_in_x_out<bool>>(va::XFunction<xt::detail::less_equal> {}, a.to_compute_variant(), b.to_compute_variant());
}
