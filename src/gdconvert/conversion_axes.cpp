#include "conversion_axes.h"

#include <cstdint>                                  // for int64_t
#include <cstddef>                                   // for ptrdiff_t
#include <stdexcept>                                 // for runtime_error
#include <vector>                                    // for vector
#include "conversion_range.h"                        // for packed_as_array
#include "godot_cpp/variant/packed_int32_array.hpp"  // for PackedInt32Array
#include "godot_cpp/variant/packed_int64_array.hpp"  // for PackedInt64Array
#include "godot_cpp/variant/variant.hpp"             // for Variant

va::Axes variant_to_axes(const Variant& variant) {
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
