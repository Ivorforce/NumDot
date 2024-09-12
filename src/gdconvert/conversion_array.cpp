#include "conversion_array.h"

#include <cmath>                                      // for double_t, float_t
#include <cmath>                                       // for double_t, float_t
#include <cstddef>                                     // for size_t
#include <cstdint>                                     // for int64_t, int32_t
#include <memory>                                      // for allocator, sha...
#include <stdexcept>                                   // for runtime_error
#include <vector>                                      // for vector
#include "godot_cpp/classes/object.hpp"                // for Object
#include "godot_cpp/core/object.hpp"                   // for Object::cast_to
#include "godot_cpp/variant/packed_byte_array.hpp"     // for PackedByteArray
#include "godot_cpp/variant/packed_float32_array.hpp"  // for PackedFloat32A...
#include "godot_cpp/variant/packed_float64_array.hpp"  // for PackedFloat64A...
#include "godot_cpp/variant/packed_int32_array.hpp"    // for PackedInt32Array
#include "godot_cpp/variant/packed_int64_array.hpp"    // for PackedInt64Array
#include "godot_cpp/variant/variant.hpp"               // for Variant
#include "godot_cpp/variant/vector2.hpp"               // for Vector2
#include "godot_cpp/variant/vector2i.hpp"              // for Vector2i
#include "godot_cpp/variant/vector3.hpp"               // for Vector3
#include "godot_cpp/variant/vector3i.hpp"              // for Vector3i
#include "godot_cpp/variant/vector4.hpp"               // for Vector4
#include "godot_cpp/variant/vector4i.hpp"              // for Vector4i
#include "ndarray.h"                                   // for NDArray
#include "xtensor/xadapt.hpp"                          // for adapt
#include "xtensor/xarray.hpp"                          // for xarray_container
#include "xtensor/xbuffer_adaptor.hpp"                 // for no_ownership
#include "xtensor/xshape.hpp"                          // for static_shape
#include "xtensor/xstorage.hpp"                        // for svector, uvector
#include "xtensor/xtensor_forward.hpp"                 // for xarray
#include "xtl/xiterator_base.hpp"                      // for operator+

template <typename C, typename T>
va::VArray packed_as_xarray(const T shape_array) {
    uint64_t size = shape_array.size();

    xt::static_shape<std::size_t, 1> shape_of_shape = { size };

    auto store = std::make_shared<xt::xarray<C>>(
        xt::xarray<C>(xt::adapt(shape_array.ptr(), size, xt::no_ownership(), shape_of_shape))
    );

    return va::from_store(store);
}

va::VArray variant_as_array(const Variant array) {
    auto type = array.get_type();

    // TODO A bunch of interesting types are still missing
    switch (type) {
        case Variant::OBJECT:
            if (auto ndarray = Object::cast_to<NDArray>(array)) {
                return ndarray->array;
            }
            break;
        case Variant::INT: {
            auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(int64_t(array)));
            return va::from_store(store);
        }
        case Variant::FLOAT: {
            auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(double_t(array)));
            return va::from_store(store);
        }
        case Variant::PACKED_BYTE_ARRAY:
            return packed_as_xarray<uint8_t>(PackedByteArray(array));
        case Variant::PACKED_INT32_ARRAY:
            return packed_as_xarray<int32_t>(PackedInt32Array(array));
        case Variant::PACKED_INT64_ARRAY:
            return packed_as_xarray<int64_t>(PackedInt64Array(array));
        case Variant::PACKED_FLOAT32_ARRAY:
            return packed_as_xarray<float_t>(PackedFloat32Array(array));
        case Variant::PACKED_FLOAT64_ARRAY:
            return packed_as_xarray<double_t>(PackedFloat64Array(array));
        case Variant::VECTOR2I: {
            auto vector = Vector2i(array);
            auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(
                { int64_t(vector.x), int64_t(vector.y) }
            ));
            return va::from_store(store);
        }
        case Variant::VECTOR3I: {
            auto vector = Vector3i(array);
            auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(
                { int64_t(vector.x), int64_t(vector.y), int64_t(vector.z) }
            ));
            return va::from_store(store);
        }
        case Variant::VECTOR4I: {
            auto vector = Vector4i(array);
            auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(
                { int64_t(vector.x), int64_t(vector.y), int64_t(vector.z), int64_t(vector.w) }
            ));
            return va::from_store(store);
        }
        case Variant::VECTOR2: {
            auto vector = Vector2(array);
            auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(
                { double_t(vector.x), double_t(vector.y) }
            ));
            return va::from_store(store);
        }
        case Variant::VECTOR3: {
            auto vector = Vector3(array);
            auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(
                { double_t(vector.x), double_t(vector.y), double_t(vector.z) }
            ));
            return va::from_store(store);
        }
        case Variant::VECTOR4: {
            auto vector = Vector4(array);
            auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(
                { double_t(vector.x), double_t(vector.y), double_t(vector.z), double_t(vector.w) }
            ));
            return va::from_store(store);
        }

        default:
            break;
    }

    // Try float first. Int may be more lossy.
    if (Variant::can_convert(type, Variant::Type::FLOAT)) {
        auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(double_t(array)));
        return va::from_store(store);
    }
    if (Variant::can_convert(type, Variant::Type::INT)) {
        auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(int64_t(array)));
        return va::from_store(store);
    }

    // TODO Godot will happily convert every number to float.
    // We should manually adapt and look through Array to find what its parts are.
    if (Variant::can_convert(type, Variant::Type::PACKED_FLOAT64_ARRAY)) {
        return packed_as_xarray<double_t>(PackedFloat64Array(array));
    }
    if (Variant::can_convert(type, Variant::Type::PACKED_INT64_ARRAY)) {
        return packed_as_xarray<int64_t>(PackedInt64Array(array));
    }

    throw std::runtime_error("Unsupported type");
}

Array xtvariant_to_godot_array(const va::VArray &array) {
    Array godot_array = Array();

    std::visit([&godot_array](auto carray){
        godot_array.resize(carray.size());
        auto start = carray.begin();

        for (size_t i = 0; i < carray.size(); ++i) {
            godot_array[i] = *(start + i);
        }
    }, array.to_compute_variant());

    return godot_array;
}
