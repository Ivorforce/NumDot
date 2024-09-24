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
        case Variant::INT:
            return std::vector { static_cast<std::ptrdiff_t>(static_cast<int64_t>(variant)) };
        case Variant::PACKED_INT32_ARRAY:
            return packed_as_array<std::vector<std::ptrdiff_t>>(PackedInt32Array(variant));
        case Variant::PACKED_INT64_ARRAY:
            return packed_as_array<std::vector<std::ptrdiff_t>>(PackedInt64Array(variant));
        case Variant::ARRAY: {
            const Array axes_array = variant;
            auto axes = va::Axes(axes_array.size());
            for (int64_t i = 0; i < axes_array.size(); i++) {
                const Variant& element = axes_array[i];
                if (element.get_type() != Variant::INT)
                    throw std::runtime_error("Axis must be an integer");
                axes[i] = static_cast<int64_t>(element);
            }
            return axes;
        }
        default:
            break;
    }

    throw std::runtime_error("Variant cannot be converted to a shape.");
}
