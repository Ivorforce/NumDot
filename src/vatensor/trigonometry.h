#ifndef TRIGONOMETRY_H
#define TRIGONOMETRY_H

#include "auto_defines.h"
#include "varray.h"

namespace va {
    void sin(VArrayTarget target, const VArray& array);
    void cos(VArrayTarget target, const VArray& array);
    void tan(VArrayTarget target, const VArray& array);

    void asin(VArrayTarget target, const VArray& array);
    void acos(VArrayTarget target, const VArray& array);
    void atan(VArrayTarget target, const VArray& array);
    void atan2(VArrayTarget target, const VArray& x1, const VArray& x2);

    void sinh(VArrayTarget target, const VArray& array);
    void cosh(VArrayTarget target, const VArray& array);
    void tanh(VArrayTarget target, const VArray& array);

    void asinh(VArrayTarget target, const VArray& array);
    void acosh(VArrayTarget target, const VArray& array);
    void atanh(VArrayTarget target, const VArray& array);
}

#endif //TRIGONOMETRY_H
