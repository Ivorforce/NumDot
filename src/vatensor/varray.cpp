#include "varray.h"

#include <cstddef>                         // for size_t
#include <type_traits>                     // for decay_t
#include "xtensor/xstrided_view_base.hpp"  // for strided_view_args

va::DType va::VArray::dtype() const {
    return DType(store.index());
}

size_t va::VArray::size() const {
    return std::visit([](auto&& carray) { return carray.size(); }, to_compute_variant());
}

size_t va::VArray::dimension() const {
    return std::visit([](auto&& carray) { return carray.dimension(); }, to_compute_variant());
}

va::VArray va::VArray::slice(const xt::xstrided_slice_vector &slices) const {
    return std::visit([slices, this](auto &store) -> VArray {
        xt::detail::strided_view_args<xt::detail::no_adj_strides_policy> args;
        args.fill_args(
            shape,
            strides,
            offset,
            layout,
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
    }, store);
}

va::ComputeVariant va::VArray::to_compute_variant() const {
    return std::visit([this](const auto& store) -> ComputeVariant {
        return va::to_compute_variant(store, *this);
    }, store);
}

size_t va::VArray::size_of_array_in_bytes() const {
    return std::visit([](auto&& carray){
        using V = typename std::decay_t<decltype(carray)>::value_type;
        return carray.size() * sizeof(V);
    }, to_compute_variant());
}
