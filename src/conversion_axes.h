#ifndef CONVERSION_AXES_H
#define CONVERSION_AXES_H

#include <godot_cpp/variant/variant.hpp>

#include "varray.h"
#include "vcompute.h"

template <typename T, typename Packed>
static T packed_as_array(Packed packed) {
    T axes;
    axes.assign(packed.ptr(), packed.ptr() + packed.size());
    return axes;
}

static va::Axes variant_to_axes(Variant variant) {
    const auto type = variant.get_type();

    switch (type) {
        case Variant::NIL:
            return nullptr;
        case Variant::INT:
            return std::vector { static_cast<std::ptrdiff_t>(static_cast<int64_t>(variant)) };
        case Variant::PACKED_INT32_ARRAY:
            return packed_as_array<std::vector<std::ptrdiff_t>>(PackedInt32Array(variant));
        case Variant::PACKED_INT64_ARRAY:
            return packed_as_array<std::vector<std::ptrdiff_t>>(PackedInt64Array(variant));
        default:
            break;
    }

    // TODO Godot will probably convert float to int. We should check.
    if (Variant::can_convert(type, Variant::Type::PACKED_INT32_ARRAY)) {
        return packed_as_array<std::vector<std::ptrdiff_t>>(PackedInt32Array(variant));
    }

    throw std::runtime_error("Variant cannot be converted to a shape.");
}

#endif //CONVERSION_AXES_H
