#include "conversion_ints.h"

template <typename C, typename T>
T variant_as_ints_(const Variant& variant) {
    switch (variant.get_type()) {
        case Variant::OBJECT: {
            if (auto ndarray = Object::cast_to<NDArray>(variant)) {
                // TODO
                // target = ndarray->array;
                throw std::runtime_error("Unsupported type");
            }
            break;
        }
        case Variant::ARRAY: {
            const Array axes_array = variant;
            auto values = T(axes_array.size());
            for (int64_t i = 0; i < axes_array.size(); i++) {
                const Variant& element = axes_array[i];
                if (element.get_type() != Variant::INT)
                    throw std::runtime_error("Axis must be an integer");
                values[i] = static_cast<int64_t>(element);
            }
            return values;
        }
        case Variant::INT:
            return { C(static_cast<int64_t>(variant)) };
        case Variant::PACKED_BYTE_ARRAY:
            return packed_as_array<T>(PackedByteArray(variant));
        case Variant::PACKED_INT32_ARRAY:
            return packed_as_array<T>(PackedInt32Array(variant));
        case Variant::PACKED_INT64_ARRAY:
            return packed_as_array<T>(PackedInt64Array(variant));
        case Variant::VECTOR2I: {
            auto vector = Vector2i(variant);
            return { C(vector.x), C(vector.y) };
        }
        case Variant::VECTOR3I: {
            auto vector = Vector3i(variant);
            return { C(vector.x), C(vector.y), C(vector.z) };
        }
        case Variant::VECTOR4I: {
            auto vector = Vector4i(variant);
            return { C(vector.x), C(vector.y), C(vector.z), C(vector.w) };
        }
        default:
            break;
    }

    throw std::runtime_error("Unsupported type");
}

std::vector<size_t> variant_to_shape(const Variant &shape) {
    return variant_as_ints_<size_t, std::vector<size_t>>(shape);
}

va::strides_type variant_to_strides(const Variant &shape) {
    return variant_as_ints_<std::ptrdiff_t, va::strides_type>(shape);
}

va::Axes variant_as_axes(const Variant &shape) {
    return variant_as_ints_<std::ptrdiff_t, va::Axes>(shape);
}
