#ifndef VMATH_H
#define VMATH_H

#include "varray.h"

namespace va {
    VArray add(const VArray& a, const VArray& b);
    VArray subtract(const VArray& a, const VArray& b);
    VArray multiply(const VArray& a, const VArray& b);
    VArray divide(const VArray& a, const VArray& b);
    VArray remainder(const VArray& a, const VArray& b);
    VArray pow(const VArray& a, const VArray& b);

    VArray minimum(const VArray& a, const VArray& b);
    VArray maximum(const VArray& a, const VArray& b);

    VArray sign(const VArray& array);
    VArray abs(const VArray& array);
    VArray sqrt(const VArray& array);
    VArray exp(const VArray& array);
    VArray log(const VArray& array);

    VArray rad2deg(const VArray& array);
    VArray deg2rad(const VArray& array);
}

#endif //MATH_H
