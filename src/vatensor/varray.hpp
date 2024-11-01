#ifndef VARRAY_H
#define VARRAY_H

#include <cmath>                           // for double_t, float_t
#include <complex>
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
#include "vstrided_view.hpp"

namespace va {
    // We should be using the same default types as xarray does, so we know for sure the ones we create /
    //  pass around are the ones we need in the end.
    using size_type = std::size_t;
    // Refer to xarray
    using shape_type = xt::dynamic_shape<size_type>;
    // Refer to xarray xcontainer_inner_types.
    using strides_type = xt::get_strides_t<shape_type>;
    using axes_type = strides_type;

    template<typename T>
    using array_case = xt::xarray<T, xt::layout_type::row_major>;

    using ArrayVariant = std::variant<
        array_case<bool>,
        array_case<float_t>,
        array_case<double_t>,
        array_case<std::complex<float_t>>,
        array_case<std::complex<double_t>>,
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
        Complex64,
        Complex128,
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
        std::complex<float_t>,
        std::complex<double_t>,
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
    template<typename T>
    using compute_case = xt::xarray_adaptor<
        xt::xbuffer_adaptor<T, xt::no_ownership, xt::detail::default_allocator_for_ptr_t<T>>,
        xt::layout_type::dynamic,
        shape_type
    >;

    using VWrite = std::variant<
        compute_case<bool*>,
        compute_case<float_t*>,
        compute_case<double_t*>,
        compute_case<std::complex<float_t>*>,
        compute_case<std::complex<double_t>*>,
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
        compute_case<const std::complex<float_t>*>,
        compute_case<const std::complex<double_t>*>,
        compute_case<const int8_t*>,
        compute_case<const int16_t*>,
        compute_case<const int32_t*>,
        compute_case<const int64_t*>,
        compute_case<const uint8_t*>,
        compute_case<const uint16_t*>,
        compute_case<const uint32_t*>,
        compute_case<const uint64_t*>
    >;

    [[nodiscard]] const shape_type& shape(const VRead& read);
    [[nodiscard]] const strides_type& strides(const VRead& read);
    [[nodiscard]] size_type offset(const VRead& read);
    [[nodiscard]] xt::layout_type layout(const VRead& read);
    [[nodiscard]] DType dtype(const VRead& read);
    [[nodiscard]] std::size_t size(const VRead& read);
    [[nodiscard]] std::size_t dimension(const VRead& read);
    [[nodiscard]] std::size_t size_of_array_in_bytes(const VRead& read);

    [[nodiscard]] VScalar to_single_value(const VRead& read);

    class VStore {
        public:
        virtual VWrite make_write(const VRead& read) = 0;

        virtual ~VStore() = default;
    };

    class VArray {
    public:
        std::shared_ptr<VStore> store;
        VRead read;
        std::optional<VWrite> write;

        [[nodiscard]] const shape_type& shape() const { return va::shape(read); }
        [[nodiscard]] const strides_type& strides() const { return va::strides(read); }
        [[nodiscard]] size_type offset() const { return va::offset(read); }
        [[nodiscard]] xt::layout_type layout() const { return va::layout(read); }

        [[nodiscard]] DType dtype() const { return va::dtype(read); }
        [[nodiscard]] std::size_t size() const { return va::size(read); }
        [[nodiscard]] std::size_t dimension() const { return va::dimension(read); }
        [[nodiscard]] std::size_t size_of_array_in_bytes() const { return va::size_of_array_in_bytes(read); }

        [[nodiscard]] VScalar to_single_value() const { return va::to_single_value(read); }

        void prepare_write();

        [[nodiscard]] std::shared_ptr<VArray> sliced(const xt::xstrided_slice_vector& slices) const;
        [[nodiscard]] VRead sliced_read(const xt::xstrided_slice_vector& slices) const;
        [[nodiscard]] VWrite sliced_write(const xt::xstrided_slice_vector& slices);

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

    template<typename V>
    static compute_case<V> make_compute(V&& ptr, const shape_type& shape, const strides_type& strides, xt::layout_type layout) {
        auto size_ = std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies());

        switch (layout) {
            case xt::layout_type::row_major:
            case xt::layout_type::column_major:
            case xt::layout_type::any:
                return xt::adapt<xt::layout_type::dynamic, V>(std::forward<V>(ptr), size_, xt::no_ownership(), shape, layout);
            default: {
                return xt::adapt<V>(std::forward<V>(ptr), size_, xt::no_ownership(), shape, strides);
            }
        }
    }

    template<typename CC>
    static auto slice_compute(CC& compute, const xt::xstrided_slice_vector& slices) {
        va::strided_view_args<xt::detail::no_adj_strides_policy> args;
        args.fill_args(
            compute.shape(),
            compute.strides(),
            compute.data_offset(),
            compute.layout(),
            slices
        );

        return make_compute(compute.data() + args.new_offset, args.new_shape, args.new_strides, args.new_layout);
    }

    template<typename S, typename VT = typename S::value_type>
    static std::shared_ptr<VArray> from_surrogate(std::shared_ptr<VStore>&& owner, const S& surrogate, const VT* data) {
        return std::make_shared<VArray>(
            VArray {
                std::forward<std::shared_ptr<VStore>>(owner),
                make_compute<const VT*>(
                    data + surrogate.data_offset(),
                    surrogate.shape(),
                    surrogate.strides(),
                    surrogate.layout()
                )
            }
        );
    }

    std::shared_ptr<VArray> from_scalar_variant(VScalar scalar);

    // For all functions returning an or assigning to an array.
    // The first case will place the array in the optional.
    // The second case will assign to the compute variant.
    using VArrayTarget = std::variant<std::shared_ptr<VArray>*, VWrite*>;

    VScalar dtype_to_variant(DType dtype);
    DType variant_to_dtype(VScalar dtype);

    std::size_t size_of_dtype_in_bytes(DType dtype);

    VScalar scalar_to_dtype(VScalar v, DType dtype);

    DType dtype_common_type(DType a, DType b);

    // TODO Can probably just be static_cast override or some such.
    template<typename V>
    V scalar_to_type(VScalar v) {
        return std::visit([](auto v) -> V {
            if constexpr (!std::is_convertible_v<decltype(v), V>) {
                throw std::runtime_error("Cannot promote in this way.");
            }
            else {
                return static_cast<V>(v);
            }
        }, v);
    }
}

#endif //VARRAY_H
