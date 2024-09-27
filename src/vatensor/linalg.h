#ifndef LINALG_H
#define LINALG_H

#include "auto_defines.h"
#include "varray.h"

namespace va {
    VScalar reduce_dot(const VArray &a, const VArray &b);
    void reduce_dot(VArrayTarget target, const VArray &a, const VArray &b, const axes_type& axes);

    void dot(VArrayTarget target, const VArray& a, const VArray& b);
    void matmul(VArrayTarget target, const VArray& a, const VArray& b);
}

#endif //LINALG_H
