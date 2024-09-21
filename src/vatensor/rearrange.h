#ifndef VA_H
#define VA_H

#include "auto_defines.h"
#include <cstddef>   // for ptrdiff_t, size_t
#include <variant>   // for visit
#include "varray.h"  // for VArray, strides_type, from_surrogate, to_strided

namespace va {
    template <typename Visitor>
    static VArray map(const Visitor& visitor, const VArray& varray) {
        return std::visit([visitor, varray](auto& store) -> VArray {
            auto strided = to_strided(store, varray);
            return from_surrogate(store, visitor(strided));
        }, varray.store);
    }

    VArray transpose(const VArray& varray, strides_type permutation);
    VArray reshape(const VArray& varray, strides_type new_shape);
    VArray swapaxes(const VArray& varray, std::ptrdiff_t a, std::ptrdiff_t b);
    VArray moveaxis(const VArray& varray, std::ptrdiff_t src, std::ptrdiff_t dst);
    VArray flip(const VArray& varray, size_t axis);
}

#endif //XV_H
