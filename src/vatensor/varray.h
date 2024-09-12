#ifndef VARRAY_H
#define VARRAY_H

#include <cmath>                        // for double_t, float_t
#include <cstddef>                      // for size_t, ptrdiff_t, nullptr_t
#include <cstdint>                      // for int16_t, int32_t, int64_t
#include <functional>                   // for multiplies
#include <memory>                       // for shared_ptr
#include <numeric>                      // for accumulate
#include <utility>                      // for move, forward
#include <variant>                      // for variant, visit
#include <vector>                       // for vector
#include "xtensor/xadapt.hpp"           // for adapt
#include "xtensor/xarray.hpp"           // for xarray_adaptor
#include "xtensor/xbuffer_adaptor.hpp"  // for no_ownership, xbuffer_adaptor
#include "xtensor/xlayout.hpp"          // for layout_type
#include "xtensor/xshape.hpp"           // for dynamic_shape
#include "xtensor/xstrided_view.hpp"    // for strided_view, xstrided_slice_...
#include "xtensor/xtensor_forward.hpp"  // for xarray

namespace va {
    using shape_type = xt::dynamic_shape<std::size_t>;
    using strides_type = xt::dynamic_shape<std::ptrdiff_t>;
    using size_type = std::size_t;

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

    using VConstant = std::variant<
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

    using GivenAxes = std::vector<std::ptrdiff_t>;

    using Axes = std::variant<
        nullptr_t,
        GivenAxes
    >;

    template <typename T>
    using array_case = xt::xarray<T>;

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

    class VArray {
    public:
        StoreVariant store;
        shape_type shape;
        strides_type strides;
        size_type offset;
        xt::layout_type layout;

        [[nodiscard]] DType dtype() const;
        [[nodiscard]] size_t size() const;
        [[nodiscard]] size_t dimension() const;

        // TODO Can probably change these to subscript syntax
        [[nodiscard]] VArray slice(const xt::xstrided_slice_vector &slices) const;
        void fill(VConstant value) const;
        void set_with_array(const VArray& value) const;

        [[nodiscard]] ComputeVariant to_compute_variant() const;
        [[nodiscard]] size_t size_of_array_in_bytes() const;

        [[nodiscard]] VConstant to_single_value() const;
    };

    template <typename T>
    static auto to_compute_variant(const store_case<T>& store, const VArray& varray) {
        auto shape = varray.shape;
        auto strides = varray.strides;
        auto size_ = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies());

        // return xt::adapt(store->data(), store->size(), xt::no_ownership(), store->shape(), store->strides());
        return xt::adapt(store->data() + varray.offset, size_, xt::no_ownership(), shape, strides);
    }

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

    VConstant dtype_to_variant(DType dtype);

    size_t size_of_dtype_in_bytes(DType dtype);

    VConstant constant_to_dtype(VConstant v, DType dtype);

    // TODO Can probably just be static_cast override or some such.
    template <typename V>
    V constant_to_type(VConstant v) {
        return std::visit([](auto v) { return static_cast<V>(v); }, v);
    }
}

#endif //VARRAY_H
