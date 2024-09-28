#ifndef VARRAY_H
#define VARRAY_H

#include <cmath>                           // for double_t, float_t
#include <cstddef>                         // for size_t
#include <cstdint>                         // for int16_t, int32_t, int64_t
#include <functional>                      // for multiplies
#include <initializer_list>                // for initializer_list
#include <memory>                          // for make_shared, shared_ptr
#include <numeric>                         // for accumulate
#include <optional>                        // for optional
#include <type_traits>                     // for decay_t
#include <utility>                         // for forward
#include <variant>                         // for variant, visit
#include <vector>                          // for vector
#include "xtensor/xadapt.hpp"              // for adapt, default_allocator_f...
#include "xtensor/xarray.hpp"              // for xarray_container, xarray_a...
#include "xtensor/xbuffer_adaptor.hpp"     // for no_ownership, xbuffer_adaptor
#include "xtensor/xlayout.hpp"             // for layout_type
#include "xtensor/xshape.hpp"              // for dynamic_shape
#include "xtensor/xstorage.hpp"            // for uvector
#include "xtensor/xstrided_view.hpp"       // for no_adj_strides_policy, xst...
#include "xtensor/xstrided_view_base.hpp"  // for strided_view_args
#include "xtensor/xtensor_forward.hpp"     // for xarray
#include "xtensor/xutils.hpp"              // for get_strides_t

namespace va {
    // We should be using the same default types as xarray does, so we know for sure the ones we create /
    //  pass around are the ones we need in the end.
    using size_type = std::size_t;
    // Refer to xarray
    using shape_type = xt::dynamic_shape<size_type>;
    // Refer to xarray xcontainer_inner_types.
    using strides_type = xt::get_strides_t<shape_type>;
    using axes_type = strides_type;

    template <typename T>using array_case = xt::xarray<T, xt::layout_type::row_major>;

    template <typename T>
    using store_case = std::shared_ptr<array_case<T>>;

    using VStore = std::variant<
        store_case<bool>,
        store_case<float_t>,
        store_case<double_t>,
        store_case<int8_t>,
        store_case<int16_t>,
        store_case<int32_t>,
        store_case<int64_t>,
        store_case<uint8_t>,
        store_case<uint16_t>,
        store_case<uint32_t>,
        store_case<uint64_t>
    >;

    using ArrayVariant = std::variant<
        array_case<bool>,
        array_case<float_t>,
        array_case<double_t>,
        array_case<int8_t>,
        array_case<int16_t>,
        array_case<int32_t>,
        array_case<int64_t>,
        array_case<uint8_t>,
        array_case<uint16_t>,
        array_case<uint32_t>,
        array_case<uint64_t>
    >;

    enum DType {
        Bool,
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

    using VScalar = std::variant<
        bool,
        float_t,
        double_t,
        int8_t,
        int16_t,
        int32_t,
        int64_t,
        uint8_t,
        uint16_t,
        uint32_t,
        uint64_t
    >;

    // P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A()
    template <typename T>
    using compute_case = xt::xarray_adaptor<
        xt::xbuffer_adaptor<T, xt::no_ownership, xt::detail::default_allocator_for_ptr_t<T>>,
        xt::layout_type::dynamic,
        shape_type
    >;

    using VWrite = std::variant<
        compute_case<bool*>,
        compute_case<float_t*>,
        compute_case<double_t*>,
        compute_case<int8_t*>,
        compute_case<int16_t*>,
        compute_case<int32_t*>,
        compute_case<int64_t*>,
        compute_case<uint8_t*>,
        compute_case<uint16_t*>,
        compute_case<uint32_t*>,
        compute_case<uint64_t*>
    >;

    using VRead = std::variant<
        compute_case<const bool*>,
        compute_case<const float_t*>,
        compute_case<const double_t*>,
        compute_case<const int8_t*>,
        compute_case<const int16_t*>,
        compute_case<const int32_t*>,
        compute_case<const int64_t*>,
        compute_case<const uint8_t*>,
        compute_case<const uint16_t*>,
        compute_case<const uint32_t*>,
        compute_case<const uint64_t*>
    >;

    class VArray {
    public:
        VStore store;
        shape_type shape;
        strides_type strides;
        size_type offset;
        xt::layout_type layout;

        [[nodiscard]] DType dtype() const;
        [[nodiscard]] std::size_t size() const;
        [[nodiscard]] std::size_t dimension() const;

        // TODO Can probably change these to subscript syntax
        [[nodiscard]] std::shared_ptr<VArray> slice(const xt::xstrided_slice_vector& slices) const;
        [[nodiscard]] VScalar get_scalar(const axes_type& index) const;

        [[nodiscard]] VRead compute_read() const;
        [[nodiscard]] VRead compute_read(const xt::xstrided_slice_vector& slices) const;
        [[nodiscard]] VWrite compute_write() const;
        [[nodiscard]] VWrite compute_write(const xt::xstrided_slice_vector& slices) const;
        [[nodiscard]] std::size_t size_of_array_in_bytes() const;

        [[nodiscard]] VScalar to_single_value() const;

        explicit operator bool() const;
        explicit operator int64_t() const;
        explicit operator int32_t() const;
        explicit operator int16_t() const;
        explicit operator int8_t() const;
        explicit operator uint64_t() const;
        explicit operator uint32_t() const;
        explicit operator uint16_t() const;
        explicit operator uint8_t() const;
        explicit operator double() const;
        explicit operator float() const;
    };

    template <typename V>
    static std::shared_ptr<VArray> from_scalar(const V value) {
        return std::make_shared<VArray>(VArray {
            std::make_shared<array_case<V>>(array_case<V>(value)),
            {},
            {},
            0,
            xt::layout_type::row_major
        });
    }

    std::shared_ptr<VArray> from_scalar_variant(VScalar scalar);

    template <typename V, typename S>
    static std::shared_ptr<VArray> from_surrogate(V&& store, const S& surrogate) {
        return std::make_shared<VArray>(VArray {
            std::forward<V>(store),
            surrogate.shape(),
            surrogate.strides(),
            surrogate.data_offset(),
            surrogate.layout()
        });
    }

    template <typename V>
    static std::shared_ptr<VArray> from_store(const V store) {
        return std::make_shared<VArray>(VArray {
            store,
            store->shape(),
            store->strides(),
            store->data_offset(),  // Should be 0, but you know...
            store->layout()
        });
    }

    // For all functions returning an or assigning to an array.
    // The first case will place the array in the optional.
    // The second case will assign to the compute variant.
    using VArrayTarget = std::variant<std::shared_ptr<VArray>*, VWrite*>;

    template <typename V, typename T>
    static compute_case<V> to_compute_variant(T& store, const VArray& varray) {
        auto shape = varray.shape;
        auto size_ = std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies());

        switch (varray.layout) {
            case xt::layout_type::row_major:
            case xt::layout_type::column_major:
                return xt::adapt<xt::layout_type::dynamic, V>(store->data() + varray.offset, size_, xt::no_ownership(), shape, varray.layout);
            default: {
                auto strides = varray.strides;
                return xt::adapt<V>(store->data() + varray.offset, size_, xt::no_ownership(), shape, strides);
            }
        }
    }

    template <typename V, typename T>
    static compute_case<V> to_compute_variant(T& store, const VArray& varray, const xt::xstrided_slice_vector& slices) {
        xt::detail::strided_view_args<xt::detail::no_adj_strides_policy> args;
        args.fill_args(
            varray.shape,
            varray.strides,
            varray.offset,
            varray.layout,
            slices
        );

        auto size_ = std::accumulate(args.new_shape.begin(), args.new_shape.end(), static_cast<std::size_t>(1), std::multiplies());

        switch (args.new_layout) {
            case xt::layout_type::row_major:
            case xt::layout_type::column_major:
                return xt::adapt<xt::layout_type::dynamic, V>(store->data() + args.new_offset, size_, xt::no_ownership(), args.new_shape, args.new_layout);
            default: {
                auto strides = varray.strides;
                return xt::adapt<V>(store->data() + args.new_offset, size_, xt::no_ownership(), args.new_shape, args.new_strides);
            }
        }
    }

    VScalar dtype_to_variant(DType dtype);
    DType variant_to_dtype(VScalar dtype);

    std::size_t size_of_dtype_in_bytes(DType dtype);

    VScalar scalar_to_dtype(VScalar v, DType dtype);

    DType dtype_common_type(DType a, DType b);

    // TODO Can probably just be static_cast override or some such.
    template <typename V>
    V scalar_to_type(VScalar v) {
        return std::visit([](auto v) { return static_cast<V>(v); }, v);
    }

    // For explicit V
    template <typename V, typename T>
    static store_case<V> make_store(T&& data) {
        return std::make_shared<array_case<V>>(array_case<V>(std::forward<T>(data)));
    }

    // For deducted V, from xexpressions
    template <typename T, typename V = typename std::decay_t<T>::value_type>
    static store_case<V> make_store(T&& data) {
        return std::make_shared<array_case<V>>(array_case<V>(std::forward<T>(data)));
    }

    template <typename V>
    static store_case<V> make_store(std::initializer_list<V> data) {
        return std::make_shared<array_case<V>>(array_case<V>(data));
    }
}

#endif //VARRAY_H
