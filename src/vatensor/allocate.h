#ifndef ALLOCATE_H
#define ALLOCATE_H

#include "varray.h"

namespace va {
    VArray full(DType dtype, DTypeVariant fill_value, shape_type shape);
    VArray empty(DType dtype, shape_type shape);

    VArray copy_as_dtype(const VArray& other, DType dtype);
}

#endif //ALLOCATE_H
