#ifndef REDUCE_H
#define REDUCE_H
#include "varray.h"

namespace va {
    void sum(VArrayTarget target, const VArray& array, const Axes& axes);
    void prod(VArrayTarget target, const VArray& array, const Axes& axes);
    void mean(VArrayTarget target, const VArray& array, const Axes& axes);
    void var(VArrayTarget target, const VArray& array, const Axes& axes);
    void std(VArrayTarget target, const VArray& array, const Axes& axes);
    void max(VArrayTarget target, const VArray& array, const Axes& axes);
    void min(VArrayTarget target, const VArray& array, const Axes& axes);
}

#endif //REDUCE_H
