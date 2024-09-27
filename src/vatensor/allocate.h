#ifndef ALLOCATE_H
#define ALLOCATE_H

#include "auto_defines.h"
#include "varray.h"

namespace va {
    VArray full(VScalar fill_value, shape_type shape);
    VArray empty(DType dtype, shape_type shape);

    VArray copy_as_dtype(const VArray& other, DType dtype);
}

#endif //ALLOCATE_H
