#include "conversion_array.h"

#include <vatensor/allocate.h>                         // for empty
#include <cmath>                                       // for double_t, float_t
#include <cstddef>                                     // for size_t
#include <cstdint>                                     // for int32_t, int64_t
#include <memory>                                      // for allocator, sha...
#include <stdexcept>                                   // for runtime_error
#include <tuple>                                       // for tuple
#include <utility>                                     // for move
#include <vector>                                      // for vector
#include "godot_cpp/classes/object.hpp"                // for Object
#include "godot_cpp/core/defs.hpp"                     // for real_t
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
#include "xtensor/xlayout.hpp"                         // for layout_type
#include "xtensor/xshape.hpp"                          // for static_shape
#include "xtensor/xstorage.hpp"                        // for svector, uvector
#include "xtensor/xstrided_view.hpp"                   // for xstrided_slice...
#include "xtensor/xtensor_forward.hpp"                 // for xarray
#include "xtl/xiterator_base.hpp"                      // for operator+


va::VArray array_as_varray(const Array& array) {
    va::shape_type shape;
    va::DType dtype = va::DTypeMax;
    std::vector<Array> current_dim_arrays = { array };

    while (true) {
        size_t current_dim_size = current_dim_arrays[0].size();

        for (const auto& array : current_dim_arrays) {
            // Sizes are the same.
            if (current_dim_size == array.size()) { continue; }

            // We defined the size before, this element can broadcast.
            if (array.size() == 1 && current_dim_size >= 1) { continue; }

            // We broadcasted before, this element defines the size.
            if (current_dim_size == 1 && array.size() > 1) {
                current_dim_size = array.size();
                continue;
            }

            throw std::runtime_error("array has an inhomogenous shape");
        }

        shape.push_back(current_dim_size);
        std::vector<Array> next_dim_arrays;

        for (const Array& array : current_dim_arrays) {
            for (int i = 0; i < array.size(); ++i) {
                const Variant& element = array[i];
                switch (element.get_type()) {
                    case Variant::ARRAY:
                        next_dim_arrays.push_back(element);
                        break;
                    case Variant::FLOAT:
                        dtype = va::dtype_common_type(dtype, va::Float64);
                        break;
                    case Variant::INT:
                        dtype = va::dtype_common_type(dtype, va::Int64);
                        break;
                    case Variant::BOOL:
                        dtype = va::dtype_common_type(dtype, va::Bool);
                        break;
                    default:
                        throw std::runtime_error("unsupported array type");
                }
            }
        }

        if (next_dim_arrays.empty()) break;

        current_dim_arrays = std::move(next_dim_arrays);
    }

    if (dtype == va::DTypeMax) dtype = va::Float64; // Default dtype

    va::VArray varray = va::empty(dtype, shape);
    std::vector<std::tuple<xt::xstrided_slice_vector, Variant>> next = { std::make_tuple(xt::xstrided_slice_vector {}, array) };

    while (!next.empty()) {
        auto [idx, var] = std::move(next.back());
        next.pop_back();

        switch (var.get_type()) {
            case Variant::ARRAY: {
                const Array array = var;
                for (size_t i = 0; i < array.size(); ++i) {
                    auto new_idx = idx;
                    new_idx.emplace_back(i);
                    next.push_back({new_idx, array[i]});
                }
                break;
            }
            case Variant::FLOAT:
                varray.slice(idx).fill(static_cast<double_t>(var));
                break;
            case Variant::INT:
                varray.slice(idx).fill(static_cast<int64_t>(var));
                break;
            default:
                throw std::runtime_error("unsupported array type");
        }
    }

    return varray;
}

template <typename C, typename T>
va::VArray packed_as_xarray(const T shape_array) {
    auto size = static_cast<std::size_t>(shape_array.size());

    xt::static_shape<std::size_t, 1> shape_of_shape = { size };

    auto store = std::make_shared<xt::xarray<C>>(
        xt::xarray<C>(xt::adapt(shape_array.ptr(), size, xt::no_ownership(), shape_of_shape))
    );

    return va::from_store(store);
}

va::VArray variant_as_array(const Variant& array) {
    switch (array.get_type()) {
        case Variant::OBJECT:
            if (auto ndarray = Object::cast_to<NDArray>(array)) {
                return ndarray->array;
            }
            break;
        case Variant::ARRAY: {
            return array_as_varray(array);
        }
        case Variant::BOOL: {
            return va::from_store(std::make_shared<xt::xarray<bool>>(xt::xarray<bool>(array)));
        }
        case Variant::INT: {
            return va::from_store(std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(array)));
        }
        case Variant::FLOAT: {
            return va::from_store(std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(array)));
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
            return va::from_store(std::make_shared<xt::xarray<int32_t>>(xt::xarray<int32_t>(
                { vector.x, vector.y }
            )));
        }
        case Variant::VECTOR3I: {
            auto vector = Vector3i(array);
            return va::from_store(std::make_shared<xt::xarray<int32_t>>(xt::xarray<int32_t>(
                { vector.x, vector.y, vector.z }
            )));
        }
        case Variant::VECTOR4I: {
            auto vector = Vector4i(array);
            return va::from_store(std::make_shared<xt::xarray<int32_t>>(xt::xarray<int32_t>(
                { vector.x, vector.y, vector.z, vector.w }
            )));
        }
        case Variant::VECTOR2: {
            auto vector = Vector2(array);
            return va::from_store(std::make_shared<xt::xarray<real_t>>(xt::xarray<real_t>(
                { vector.x, vector.y }
            )));
        }
        case Variant::VECTOR3: {
            auto vector = Vector3(array);
            return va::from_store(std::make_shared<xt::xarray<real_t>>(xt::xarray<real_t>(
                { vector.x, vector.y, vector.z }
            )));
        }
        case Variant::VECTOR4: {
            auto vector = Vector4(array);
            return va::from_store(std::make_shared<xt::xarray<real_t>>(xt::xarray<real_t>(
                { vector.x, vector.y, vector.z, vector.w }
            )));
        }
        default:
            break;
    }

    throw std::runtime_error("Unsupported type");
}

Array varray_to_godot_array(const va::VArray &array) {
    Array godot_array = Array();

    // TODO Non-flat
    std::visit([&godot_array](auto carray){
        godot_array.resize(carray.size());
        auto start = carray.begin();

        for (size_t i = 0; i < carray.size(); ++i) {
            godot_array[i] = *(start + i);
        }
    }, array.to_compute_variant());

    return godot_array;
}
