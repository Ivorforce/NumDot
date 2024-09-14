#include "logical.h"

#include "vatensor/varray.h"            // for VArray
#include "vcompute.h"                   // for XFunction, xoperation
#include "vpromote.h"                   // for bool_in_bool_out
#include "xtensor/xlayout.hpp"          // for layout_type
#include "xtensor/xoperation.hpp"       // for logical_and, make_xfunction

using namespace va;

VArray va::logical_and(const VArray &a, const VArray &b) {
    return xoperation<promote::bool_in_bool_out>(XFunction<xt::detail::logical_and> {}, a.to_compute_variant(), b.to_compute_variant());
}

VArray va::logical_or(const VArray &a, const VArray &b) {
    return xoperation<promote::bool_in_bool_out>(XFunction<xt::detail::logical_or> {}, a.to_compute_variant(), b.to_compute_variant());
}
