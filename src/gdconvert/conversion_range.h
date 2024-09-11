#ifndef CONVERSION_RANGE_H
#define CONVERSION_RANGE_H

#include <ndrange.h>
#include <godot_cpp/variant/variant.hpp>

template <typename T, typename Packed>
T packed_as_array(Packed packed) {
    T axes;
    axes.assign(packed.ptr(), packed.ptr() + packed.size());
    return axes;
}

range_part to_range_part(const Variant& variant);

#endif //CONVERSION_RANGE_H
