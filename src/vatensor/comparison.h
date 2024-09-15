#ifndef COMPARISON_H
#define COMPARISON_H
#include "varray.h"

namespace va {
    VArray equal_to(const VArray& a, const VArray& b);
    VArray not_equal_to(const VArray& a, const VArray& b);
    VArray greater(const VArray& a, const VArray& b);
    VArray greater_equal(const VArray& a, const VArray& b);
    VArray less(const VArray& a, const VArray& b);
    VArray less_equal(const VArray& a, const VArray& b);
}

#endif //COMPARISON_H
