#ifndef VMATH_H
#define VMATH_H

#include "varray.h"

namespace va {
    void add(VArrayTarget target, const VArray& a, const VArray& b);
    void subtract(VArrayTarget target, const VArray& a, const VArray& b);
    void multiply(VArrayTarget target, const VArray& a, const VArray& b);
    void divide(VArrayTarget target, const VArray& a, const VArray& b);
    void remainder(VArrayTarget target, const VArray& a, const VArray& b);
    void pow(VArrayTarget target, const VArray& a, const VArray& b);

    void minimum(VArrayTarget target, const VArray& a, const VArray& b);
    void maximum(VArrayTarget target, const VArray& a, const VArray& b);

    void sign(VArrayTarget target, const VArray& array);
    void abs(VArrayTarget target, const VArray& array);
    void sqrt(VArrayTarget target, const VArray& array);
    void exp(VArrayTarget target, const VArray& array);
    void log(VArrayTarget target, const VArray& array);

    void rad2deg(VArrayTarget target, const VArray& array);
    void deg2rad(VArrayTarget target, const VArray& array);
}

#endif //MATH_H
