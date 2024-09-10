#ifndef VARRAY_H
#define VARRAY_H

#include "xtensor/xstrided_view.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xstrided_view.hpp"
#include <godot_cpp/variant/utility_functions.hpp>

namespace va {
    using shape_type = xt::dynamic_shape<std::size_t>;
    using strides_type = xt::dynamic_shape<std::ptrdiff_t>;
    using size_type = std::size_t;

    template <typename T>
    using array_case = xt::xarray<T>;

    template <typename T>
    using store_case = std::shared_ptr<array_case<T>>;

    using StoreVariant = std::variant<
        store_case<double_t>,
        store_case<float_t>,
        store_case<int8_t>,
        store_case<int16_t>,
        store_case<int32_t>,
        store_case<int64_t>,
        store_case<uint8_t>,
        store_case<uint16_t>,
        store_case<uint32_t>,
        store_case<uint64_t>
    >;

    struct VArray {
        StoreVariant store;
        shape_type shape;
        strides_type strides;
        size_type offset;
        xt::layout_type layout;
    };

    template <typename T>
    static auto to_compute_variant_(const store_case<T>& store, const VArray &varray) {
        auto shape = varray.shape;
        auto strides = varray.strides;
        auto size_ = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies());

        // return xt::adapt(store->data(), store->size(), xt::no_ownership(), store->shape(), store->strides());
        return xt::adapt(store->data() + varray.offset, size_, xt::no_ownership(), shape, strides);
    }

    // P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A()
    template <typename T>
    using compute_case = xt::xarray_adaptor<xt::xbuffer_adaptor<T*>,xt::layout_type::dynamic>;

    using ComputeVariant = std::variant<
        compute_case<double_t>,
        compute_case<float_t>,
        compute_case<int8_t>,
        compute_case<int16_t>,
        compute_case<int32_t>,
        compute_case<int64_t>,
        compute_case<uint8_t>,
        compute_case<uint16_t>,
        compute_case<uint32_t>,
        compute_case<uint64_t>
    >;

    using DTypeVariant = std::variant<
        double_t,
        float_t,
        int8_t,
        int16_t,
        int32_t,
        int64_t,
        uint8_t,
        uint16_t,
        uint32_t,
        uint64_t
    >;

    enum DType {
        Float32,
        Float64,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        DTypeMax
    };

    template <typename T>
    static VArray dummy() {
        return VArray {
            std::make_shared<xt::xarray<T>>(xt::xarray<T>({ 0 })),
            { 1 },
            { 1 },
            0,
            xt::layout_type::dynamic,
        };
    }

    static VArray slice(const VArray& varray, const xt::xstrided_slice_vector &slices) {
        return std::visit([slices, varray](auto &store) -> VArray {
            xt::detail::strided_view_args<xt::detail::no_adj_strides_policy> args;
            args.fill_args(
                varray.shape,
                varray.strides,
                varray.offset,
                varray.layout,
                slices
            );

            auto result = VArray{
                store,  // Implicit copy
                std::move(args.new_shape),
                std::move(args.new_strides),
                args.new_offset,
                args.new_layout
            };

            return result;
        }, varray.store);
    }

    static ComputeVariant to_compute_variant(const VArray& varray) {
        return std::visit([varray](const auto& store) -> ComputeVariant {
            return to_compute_variant_(store, varray);
        }, varray.store);
    };

    template <typename T>
    static auto to_strided(T&& store, const VArray& varray) {
        auto shape = varray.shape;
        auto strides = varray.strides;

        return xt::strided_view(
            std::forward<T>(store),
            std::move(shape),
            std::move(strides),
            varray.offset,
            varray.layout
        );
    }

    template <typename V>
    static VArray from_store(const V store) {
        return {
            store,
            store->shape(),
            store->strides(),
            0,
            xt::layout_type::dynamic
        };
    }

    template <typename V, typename S>
    static VArray from_surrogate(V&& store, const S& surrogate) {
        return {
            std::forward<V>(store),
            surrogate.shape(),
            surrogate.strides(),
            surrogate.data_offset(),
            surrogate.layout()
        };
    }

    template <typename Visitor>
    static VArray map(const Visitor& visitor, const VArray& varray) {
        return std::visit([visitor, varray](auto& store) -> VArray {
            auto strided = to_strided(store, varray);
            return from_surrogate(store, visitor(strided));
        }, varray.store);
    }

    static VArray transpose(const VArray& varray, strides_type permutation) {
        return map([permutation](auto& array) {
            return xt::transpose(
                array,
                permutation,
                xt::check_policy::full{}
            );
        }, varray);
    }

    static VArray reshape(const VArray& varray, strides_type new_shape) {
        return map([new_shape](auto& array) {
            auto new_shape_ = new_shape;
            return xt::reshape_view(array, new_shape_);
        }, varray);
    }

    static VArray swapaxes(const VArray& varray, std::ptrdiff_t a, std::ptrdiff_t b) {
        return map([a, b](auto& array) {
            return xt::swapaxes(array, a, b);
        }, varray);
    }

    static VArray moveaxis(const VArray& varray, std::ptrdiff_t src, std::ptrdiff_t dst) {
        return map([src, dst](auto& array) {
            return xt::moveaxis(array, src, dst);
        }, varray);
    }

    static VArray flip(const VArray& varray, size_t axis) {
        return map([axis](auto& array) {
            return xt::flip(array, axis);
        }, varray);
    }

    template <typename T>
    static inline DType dtype(T&& variant) {
        return DType(std::forward<T>(variant).store.index());
    }

    template <typename T>
    static inline auto shape(T&& variant) {
        return std::visit([](auto&& carray) { return carray.shape(); }, to_compute_variant(std::forward<T>(variant)));
    }

    template <typename T>
    static inline size_t size(T&& variant) {
        return std::visit([](auto&& carray) { return carray.size(); }, to_compute_variant(std::forward<T>(variant)));
    }

    template <typename T>
    static inline size_t dimension(T&& variant) {
        return std::visit([](auto&& carray) { return carray.dimension(); }, to_compute_variant(std::forward<T>(variant)));
    }

    static DTypeVariant dtype_to_variant(const DType dtype) {
        switch (dtype) {
            case DType::Float32:
                return float_t();
            case DType::Float64:
                return double_t();
            case DType::Int8:
                return int8_t();
            case DType::Int16:
                return int16_t();
            case DType::Int32:
                return int32_t();
            case DType::Int64:
                return int64_t();
            case DType::UInt8:
                return uint8_t();
            case DType::UInt16:
                return uint16_t();
            case DType::UInt32:
                return uint32_t();
            case DType::UInt64:
                return int64_t();
            case DType::DTypeMax:
                throw std::runtime_error("Invalid dtype.");
        }
    }

    static inline size_t size_of_dtype_in_bytes(const DType dtype) {
        return std::visit([](auto dtype){
            return sizeof(dtype);
        }, dtype_to_variant(dtype));
    }

    template <typename T>
    static inline size_t size_of_array_in_bytes(T&& array) {
        return std::visit([](auto&& carray){
            using V = typename std::decay_t<decltype(carray)>::value_type;
            return carray.size() * sizeof(V);
        }, to_compute_variant(array));
    }

    static VArray copy_as_dtype(const VArray& other, const DType dtype) {
        return std::visit([](auto t, auto carray) -> VArray {
            using T = decltype(t);
            return from_store(std::make_shared<xt::xarray<T>>(carray));
        }, dtype_to_variant(dtype), to_compute_variant(other));
    }

    template<typename T>
    static inline T to_single_value(const VArray& varray) {
        return std::visit([](const auto carray) {
            if (carray.size() != 1) {
                throw std::runtime_error("Expected a single element after slicing.");
            }
            return static_cast<T>(*carray.data());
            // TODO I expected this to work, but it doesn't. See https://xtensor.readthedocs.io/en/latest/indices.html#operator
            // But at least the above is a view, so no copy is made.
            // return V(array[slice]);
        }, to_compute_variant(varray));
    }

    template <typename V>
    static void set_value(const VArray& varray, V&& value) {
        return std::visit([&value](auto&& carray) {
            carray.fill(std::forward<V>(value));
        }, to_compute_variant(varray));
    }

    static void set_with_array(const VArray& array, const VArray& value) {
        return std::visit([](auto&& carray, auto&& cvalue) {
            carray.computed_assign(std::forward<decltype(cvalue)>(cvalue));
        }, to_compute_variant(array), to_compute_variant(value));
    }

    template <typename V, typename Sh>
    VArray full(const DType dtype, const V fill_value, const Sh& shape) {
        return std::visit([fill_value, shape](auto t) {
            using T = decltype(t);
            auto store = std::make_shared<xt::xarray<T>>(xt::xarray<T>::from_shape(shape));
            store->fill(fill_value);
            return from_store(store);
        }, dtype_to_variant(dtype));
    }

    template <typename Sh>
    VArray empty(const DType dtype, const Sh& shape) {
        return std::visit([shape](auto t) {
            using T = decltype(t);
            auto store = std::make_shared<xt::xarray<T>>(xt::empty<T>(shape));
            return from_store(store);
        }, dtype_to_variant(dtype));
    }

}

#endif //VARRAY_H
