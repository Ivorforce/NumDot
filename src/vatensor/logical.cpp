#include "logical.h"

#include <utility>                                      // for move
#include "vatensor/varray.h"                            // for VArray, VArra...
#include "vcompute.h"                           // for XFunction
#include "vpromote.h"                                   // for bool_in_bool_out
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xoperation.hpp"                       // for logical_and

using namespace va;

void va::logical_and(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::bool_in_bool_out>(
        XFunction<xt::detail::logical_and> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

void va::logical_or(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::bool_in_bool_out>(
        XFunction<xt::detail::logical_or> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

void va::logical_xor(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::bool_in_bool_out>(
        XFunction<xt::detail::not_equal_to> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

void va::logical_not(VArrayTarget target, const VArray& a) {
    va::xoperation_inplace<promote::bool_in_bool_out>(
        XFunction<xt::detail::logical_not> {},
        target,
        a.to_compute_variant()
    );
}
