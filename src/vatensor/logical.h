#ifndef LOGICAL_H
#define LOGICAL_H

#include "varray.h"

namespace va {
    VArray logical_and(const VArray& a, const VArray& b);
    VArray logical_or(const VArray& a, const VArray& b);
    VArray logical_not(const VArray& a);
}

#endif //LOGICAL_H
