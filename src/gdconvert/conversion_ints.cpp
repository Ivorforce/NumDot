#include "conversion_ints.h"

template <typename C, typename T>
T variant_as_ints_(const Variant& variant) {
    switch (variant.get_type()) {
        case Variant::OBJECT: {
            if (const auto ndarray = Object::cast_to<NDArray>(variant)) {
                return std::visit([](const auto& carray) -> T {
                    using V = typename std::decay_t<decltype(carray)>::value_type;

                    if constexpr (!std::is_integral_v<V>) {
                        throw std::runtime_error("incompatible dtype; must be int");
                    }

                    switch (carray.dimension()) {
                        case 0:
                            return T { C(carray(0)) };
                        case 1: {
                            T ints;
                            ints.resize(carray.size());
                            std::copy(carray.cbegin(), carray.cend(), ints.begin());
                            return ints;
                        }
                        default:
                            throw std::runtime_error("array must be zero-dimensional or one-dimensional");
                    }
                }, ndarray->array.to_compute_variant());
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

va::shape_type variant_to_shape(const Variant &variant) {
    return variant_as_ints_<size_t, std::vector<size_t>>(variant);
}

va::strides_type variant_to_axes(const Variant &variant) {
    return variant_as_ints_<std::ptrdiff_t, va::strides_type>(variant);
}
