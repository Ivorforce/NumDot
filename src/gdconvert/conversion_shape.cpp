#include "conversion_shape.h"

template <typename C, typename T>
T variant_as_ints_(const Variant& shape) {
    auto type = shape.get_type();

    switch (type) {
        case Variant::OBJECT:
            if (auto ndarray = Object::cast_to<NDArray>(shape)) {
                // TODO
                // target = ndarray->array;
                throw std::runtime_error("Unsupported type");
            }
        break;
        case Variant::INT:
            return { C(int64_t(shape)) };
        case Variant::PACKED_BYTE_ARRAY:
            return packed_as_shape<T>(PackedByteArray(shape));
        case Variant::PACKED_INT32_ARRAY:
            return packed_as_shape<T>(PackedInt32Array(shape));
        case Variant::PACKED_INT64_ARRAY:
            return packed_as_shape<T>(PackedInt64Array(shape));
        case Variant::VECTOR2I: {
            auto vector = Vector2i(shape);
            return { C(vector.x), C(vector.y) };
        }
        case Variant::VECTOR3I: {
            auto vector = Vector3i(shape);
            return { C(vector.x), C(vector.y), C(vector.z) };
        }
        case Variant::VECTOR4I: {
            auto vector = Vector4i(shape);
            return { C(vector.x), C(vector.y), C(vector.z), C(vector.w) };
        }
        default:
            break;
    }

    // TODO Godot will probably convert float to int. We should check.
    if (Variant::can_convert(type, Variant::Type::PACKED_INT32_ARRAY)) {
        return packed_as_shape<T>(PackedInt32Array(shape));
    }

    throw std::runtime_error("Unsupported type");
}

std::vector<size_t> variant_as_shape(const Variant &shape) {
    return variant_as_ints_<size_t, std::vector<size_t>>(shape);
}

va::strides_type variant_as_strides(const Variant &shape) {
    return variant_as_ints_<std::ptrdiff_t, va::strides_type>(shape);
}
