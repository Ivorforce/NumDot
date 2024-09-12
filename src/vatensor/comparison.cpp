#include "comparison.h"

#include "vatensor/varray.h"                            // for VArray
#include "vcompute.h"                                   // for XFunction
#include "vpromote.h"                                   // for identity_in_b...
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xoperation.hpp"                       // for equal_to, mak...

using namespace va;

VArray va::equal_to(const VArray &a, const VArray &b) {
    return xoperation<promote::identity_in_bool_out>(va::XFunction<xt::detail::equal_to> {}, a.to_compute_variant(), b.to_compute_variant());
}
