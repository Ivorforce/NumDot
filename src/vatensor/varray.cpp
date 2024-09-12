#include "varray.h"

#include <cstddef>     // for size_t
#include <type_traits>  // for decay_t

va::DType va::VArray::dtype() const {
    return DType(store.index());
}

size_t va::VArray::size() const {
    return std::visit([](auto&& carray) { return carray.size(); }, to_compute_variant());
}

size_t va::VArray::dimension() const {
    return std::visit([](auto&& carray) { return carray.dimension(); }, to_compute_variant());
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
