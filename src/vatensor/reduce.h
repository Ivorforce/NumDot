#ifndef REDUCE_H
#define REDUCE_H

#include "auto_defines.h"
#include "varray.h"

namespace va {
    VConstant sum(const VArray& array);
    void sum(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant prod(const VArray& array);
    void prod(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant mean(const VArray& array);
    void mean(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant var(const VArray& array);
    void var(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant std(const VArray& array);
    void std(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant max(const VArray& array);
    void max(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant min(const VArray& array);
    void min(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant norm_l0(const VArray& array);
    void norm_l0(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant norm_l1(const VArray& array);
    void norm_l1(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant norm_l2(const VArray& array);
    void norm_l2(VArrayTarget target, const VArray& array, const axes_type& axes);

    VConstant norm_linf(const VArray& array);
    void norm_linf(VArrayTarget target, const VArray& array, const axes_type& axes);

    bool all(const VArray& array);
    void all(VArrayTarget target, const VArray& array, const axes_type& axes);

    bool any(const VArray& array);
    void any(VArrayTarget target, const VArray& array, const axes_type& axes);
}

#endif //REDUCE_H
