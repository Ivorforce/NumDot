#ifndef NUMDOT_AS_ARRAY_H
#define NUMDOT_AS_ARRAY_H

#include "vatensor/auto_defines.h"
#include <godot_cpp/variant/variant.hpp>  // for Variant
#include <variant>                        // for visit
#include "godot_cpp/variant/array.hpp"    // for Array
#include "vatensor/varray.h"              // for VArray

using namespace godot;

va::VArray variant_as_array(const Variant& array);
va::VArray variant_as_array(const Variant& array, va::DType dtype, bool copy);

template <typename T>
void fill_c_array_flat(T* target, const va::VRead &array) {
    std::visit([target](auto &carray) {
        std::copy(carray.begin(), carray.end(), target);
    }, array);
}

template <typename T>
auto adapt_c_array(T&& ptr, const va::shape_type& shape) {
    const auto size = std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<>());
    return xt::adapt<xt::layout_type::dynamic, T, xt::no_ownership, va::shape_type>(
        std::forward<T>(ptr), size, xt::no_ownership(), shape, xt::layout_type::row_major
    );
}

void find_shape_and_dtype(va::shape_type& shape, va::DType &dtype, const Array& input_array);
Array varray_to_godot_array(const va::VArray& array);

#endif
