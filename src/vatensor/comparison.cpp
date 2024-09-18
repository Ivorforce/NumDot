#include "comparison.h"

#include <utility>                                       // for move
#include "vatensor/varray.h"                             // for VArray, VArr...
#include "vcompute.h"                            // for XFunction
#include "vpromote.h"                                    // for common_num_i...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xoperation.hpp"                        // for equal_to

using namespace va;

void va::equal_to(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::common_in_bool_out>(
        va::XFunction<xt::detail::equal_to> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

void va::not_equal_to(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::common_in_bool_out>(
        va::XFunction<xt::detail::not_equal_to> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

void va::greater(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
        va::XFunction<xt::detail::greater> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

void va::greater_equal(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
        va::XFunction<xt::detail::greater_equal> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

void va::less(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
        va::XFunction<xt::detail::less> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}

void va::less_equal(VArrayTarget target, const VArray& a, const VArray& b) {
    va::xoperation_inplace<promote::common_num_in_x_out<bool>>(
        va::XFunction<xt::detail::less_equal> {},
        target,
        a.to_compute_variant(),
        b.to_compute_variant()
    );
}
