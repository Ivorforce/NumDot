#ifndef COMPARISON_H
#define COMPARISON_H
#include "varray.h"

namespace va {
    void equal_to(VArrayTarget target, const VArray& a, const VArray& b);
    void not_equal_to(VArrayTarget target, const VArray& a, const VArray& b);
    void greater(VArrayTarget target, const VArray& a, const VArray& b);
    void greater_equal(VArrayTarget target, const VArray& a, const VArray& b);
    void less(VArrayTarget target, const VArray& a, const VArray& b);
    void less_equal(VArrayTarget target, const VArray& a, const VArray& b);
}

#endif //COMPARISON_H
