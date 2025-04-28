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
#include "dtype.hpp"
#include "xtensor/containers/xadapt.hpp"              // for adapt, default_allocator_f...
#include "xtensor/containers/xarray.hpp"              // for xarray_container, xarray_a...
#include "xtensor/containers/xbuffer_adaptor.hpp"     // for no_ownership, xbuffer_adaptor
#include "xtensor/core/xlayout.hpp"             // for layout_type
#include "xtensor/core/xshape.hpp"              // for dynamic_shape
#include "xtensor/containers/xstorage.hpp"            // for uvector
#include "xtensor/views/xstrided_view.hpp"       // for no_adj_strides_policy, xst...
#include "xtensor/views/xstrided_view_base.hpp"  // for strided_view_args
#include "xtensor/core/xtensor_forward.hpp"     // for xarray
#include "xtensor/utils/xutils.hpp"              // for get_strides_t
#include "vstrided_view.hpp"

namespace va {
    // P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A()
    template<typename T>
    using compute_case = xt::xarray_adaptor<
        xt::xbuffer_adaptor<T, xt::no_ownership, xt::detail::default_allocator_for_ptr_t<T>>,
        xt::layout_type::dynamic,
        shape_type
    >;

    using VData = std::variant<
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

    [[nodiscard]] const shape_type& shape(const VData& read);
    [[nodiscard]] const strides_type& strides(const VData& read);
    [[nodiscard]] size_type offset(const VData& read);
    [[nodiscard]] xt::layout_type layout(const VData& read);
    [[nodiscard]] static constexpr DType dtype(const VData& read) { return static_cast<DType>(read.index()); }
    [[nodiscard]] std::size_t size(const VData& read);
    [[nodiscard]] std::size_t dimension(const VData& read);
    [[nodiscard]] std::size_t size_of_array_in_bytes(const VData& read);
    [[nodiscard]] static bool is_contiguous(const VData& read) {
        const auto layout = va::layout(read);
        return layout == xt::layout_type::any || layout == xt::layout_type::row_major;
    }

    VData sliced_data(const VData& data, const xt::xstrided_slice_vector& slices);
    VData sliced_data(const VData& data, const xt::xstrided_slice<std::ptrdiff_t>& slice, std::ptrdiff_t axis);
    [[nodiscard]] VScalar to_single_value(const VData& read);

    class VStore {
        public:
        virtual void* data() = 0;
        virtual DType dtype() = 0;
        virtual std::size_t size() = 0;
        virtual void prepare_write(VData& data, std::ptrdiff_t data_offset) {}
        virtual ~VStore() = default;
    };

    class VStoreAllocator {
    public:
        virtual std::shared_ptr<VStore> allocate(DType dtype, std::size_t count) = 0;
        virtual ~VStoreAllocator() = default;
    };

    class VArray {
    public:
        std::shared_ptr<VStore> store;
        VData data;
        // This is not stored by data (xarray_adaptor.data_offset() is 0, because the pointer is pre-offset).
        std::ptrdiff_t data_offset;

        [[nodiscard]] const shape_type& shape() const { return va::shape(data); }
        [[nodiscard]] const strides_type& strides() const { return va::strides(data); }
        [[nodiscard]] size_type offset() const { return va::offset(data); }
        [[nodiscard]] xt::layout_type layout() const { return va::layout(data); }

        [[nodiscard]] DType dtype() const { return va::dtype(data); }
        [[nodiscard]] std::size_t size() const { return va::size(data); }
        [[nodiscard]] std::size_t dimension() const { return va::dimension(data); }
        [[nodiscard]] std::size_t is_full_view() const { return dtype() == store->dtype() && data_offset == 0 && size() == store->size(); }
        [[nodiscard]] bool is_contiguous() const { return va::is_contiguous(data); }

        [[nodiscard]] VScalar to_single_value() const { return va::to_single_value(data); }

        void prepare_write() { store->prepare_write(data, data_offset); }

        [[nodiscard]] std::shared_ptr<VArray> sliced(const xt::xstrided_slice_vector& slices) const;
        [[nodiscard]] std::shared_ptr<VArray> sliced(const xt::xstrided_slice<std::ptrdiff_t>& slice, std::ptrdiff_t axis) const;
        [[nodiscard]] VData sliced_data(const xt::xstrided_slice_vector& slices) const {
            return va::sliced_data(data, slices);
        }
        [[nodiscard]] VData sliced_data(const xt::xstrided_slice<std::ptrdiff_t>& slice, std::ptrdiff_t axis) const {
            return va::sliced_data(data, slice, axis);
        }

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
    static auto slice_compute(CC& compute, const xt::xstrided_slice_vector& slices, std::ptrdiff_t& new_offset) {
		using V = typename std::decay_t<decltype(compute)>::value_type;

        va::strided_view_args<xt::detail::no_adj_strides_policy> args;
        args.fill_args(
            compute.shape(),
            compute.strides(),
            compute.data_offset(),
            compute.layout(),
            slices
        );

        new_offset = static_cast<std::ptrdiff_t>(args.new_offset);
        return make_compute(const_cast<V*>(compute.data()) + args.new_offset, args.new_shape, args.new_strides, args.new_layout);
    }

    template<typename CC>
    static auto slice_compute(CC& compute, const xt::xstrided_slice<std::ptrdiff_t>& slice, std::ptrdiff_t axis, std::ptrdiff_t& new_offset) {
		using V = typename std::decay_t<decltype(compute)>::value_type;

        va::strided_view_args<xt::detail::no_adj_strides_policy> args;
        args.fill_args(
            compute.shape(),
            compute.strides(),
            compute.data_offset(),
            compute.layout(),
            slice,
            axis
        );

        new_offset = static_cast<std::ptrdiff_t>(args.new_offset);
        return make_compute(const_cast<V*>(compute.data()) + args.new_offset, args.new_shape, args.new_strides, args.new_layout);
    }

    template<typename S, typename VT = typename S::value_type>
    static VData compute_from_surrogate(const S& surrogate, VT* data) {
        return make_compute<VT*>(
            data + surrogate.data_offset(),
            surrogate.shape(),
            surrogate.strides(),
            surrogate.layout()
        );
    }

    template<typename S, typename VT = typename S::value_type>
    static std::shared_ptr<VArray> from_surrogate(const VArray& varray, const S& surrogate, VT* data) {
        return std::make_shared<VArray>(
            VArray {
                std::shared_ptr(varray.store),
                compute_from_surrogate(surrogate, data),
                varray.data_offset + static_cast<std::ptrdiff_t>(surrogate.data_offset())
            }
        );
    }

    // For all functions returning an or assigning to an array.
    // The first case will place the array in the optional.
    // The second case will assign to the compute variant.
    using VArrayTarget = std::variant<std::shared_ptr<VArray>*, VData*>;
}

#endif //VARRAY_H
