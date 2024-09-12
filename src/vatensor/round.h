#ifndef ROUND_H
#define ROUND_H

#include "varray.h"

namespace va {
    VArray ceil(const VArray& array);
    VArray floor(const VArray& array);
    VArray trunc(const VArray& array);
    VArray round(const VArray& array);
    VArray nearbyint(const VArray& array);
}

#endif //ROUND_H
