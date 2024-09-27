#ifndef NUMDOT_VASSIGN_H
#define NUMDOT_VASSIGN_H

#include "xtensor/xview.hpp"  // for strided_view_args
#include "varray.h"

namespace va {
    // computed_assign on containers doesn't assign data, it tries to assign to the whole container.
    // This is basically view_semantic's computed_assign.
    template <typename T, typename E>
    inline void broadcasting_assign(xt::xexpression<T>& t, const xt::xexpression<E>& e) {
        xt::assert_compatible_shape(t, e);
        xt::assign_data(t, e, xt::detail::get_rhs_triviality(e.derived_cast()));
    }

    void assign(VWrite& array, const VRead& value);
    void assign_nonoverlapping(VWrite& array, const ArrayVariant& value);
    void assign(VWrite& array, VScalar value);

    void assign(VArrayTarget target, VScalar value);
}

#endif //NUMDOT_VASSIGN_H
