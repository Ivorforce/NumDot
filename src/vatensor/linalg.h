#ifndef LINALG_H
#define LINALG_H

#include "auto_defines.h"
#include "varray.h"

namespace va {
    void dot(VArrayTarget target, const VArray& a, const VArray& b, const Axes& axes);
}

#endif //LINALG_H
