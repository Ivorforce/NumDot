#ifndef REDUCE_H
#define REDUCE_H
#include "varray.h"

namespace va {
    VArray sum(const VArray& array, const Axes& axes);
    VArray prod(const VArray& array, const Axes& axes);
    VArray mean(const VArray& array, const Axes& axes);
    VArray var(const VArray& array, const Axes& axes);
    VArray std(const VArray& array, const Axes& axes);
    VArray max(const VArray& array, const Axes& axes);
    VArray min(const VArray& array, const Axes& axes);
}

#endif //REDUCE_H
