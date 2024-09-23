#ifndef NUMDOT_VASSIGN_H
#define NUMDOT_VASSIGN_H

#include "xtensor/xview.hpp"  // for strided_view_args
#include "varray.h"

namespace va {
    void assign(ComputeVariant& array, const ComputeVariant& value);
    void assign(ComputeVariant& array, const ArrayVariant& value);
    void assign(ComputeVariant& array, VConstant value);
}

#endif //NUMDOT_VASSIGN_H
