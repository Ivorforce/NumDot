//
// Created by Lukas Tenbrink on 12.09.24.
//

#include "rearrange.h"

#include "xtensor/xlayout.hpp"        // for layout_type
#include "xtensor/xmanipulation.hpp"  // for flip, full, moveaxis, swapaxes
#include "xtensor/xstrided_view.hpp"  // for reshape_view
#include "vatensor//varray.h"          // for VArray, strides_type

using namespace va;

VArray va::transpose(const VArray &varray, strides_type permutation) {
    return map([permutation](auto& array) {
        return xt::transpose(
            array,
            permutation,
            xt::check_policy::full{}
        );
    }, varray);
}

VArray va::reshape(const VArray &varray, strides_type new_shape) {
    return map([new_shape](auto& array) {
        auto new_shape_ = new_shape;
        return xt::reshape_view(array, new_shape_);
    }, varray);
}

VArray va::swapaxes(const VArray &varray, std::ptrdiff_t a, std::ptrdiff_t b) {
    return map([a, b](auto& array) {
        return xt::swapaxes(array, a, b);
    }, varray);
}

VArray va::moveaxis(const VArray &varray, std::ptrdiff_t src, std::ptrdiff_t dst) {
    return map([src, dst](auto& array) {
        return xt::moveaxis(array, src, dst);
    }, varray);
}

VArray va::flip(const VArray &varray, size_t axis) {
    return map([axis](auto& array) {
        return xt::flip(array, axis);
    }, varray);
}
