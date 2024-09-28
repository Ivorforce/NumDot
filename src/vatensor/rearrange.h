#ifndef VA_H
#define VA_H

#include "auto_defines.h"

#include <cstddef>      // for ptrdiff_t, size_t
#include <type_traits>  // for decay_t
#include <variant>      // for visit
#include "varray.h"     // for VArray, strides_type, axes_type, from_surrogate

namespace va {
    template <typename Visitor>
    static VArray map(const Visitor& visitor, const VArray& varray) {
        return std::visit([visitor, varray](auto& store) -> VArray {
        using V = typename std::decay_t<decltype(store)>::element_type::value_type;
            auto read = to_compute_variant<const V*>(store, varray);
            return from_surrogate(store, visitor(read));
        }, varray.store);
    }

    VArray transpose(const VArray& varray, strides_type permutation);
    VArray reshape(const VArray& varray, strides_type new_shape);
    VArray swapaxes(const VArray& varray, std::ptrdiff_t a, std::ptrdiff_t b);
    VArray moveaxis(const VArray& varray, std::ptrdiff_t src, std::ptrdiff_t dst);
    VArray flip(const VArray& varray, std::size_t axis);
    VArray join_axes_into_last_dimension(const VArray& varray, axes_type axes);
}

#endif //XV_H
