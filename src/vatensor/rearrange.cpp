//
// Created by Lukas Tenbrink on 12.09.24.
//

#include "rearrange.h"

#include <algorithm>                  // for stable_partition
#include <functional>                 // for multiplies
#include <numeric>                    // for accumulate, iota
#include <set>                        // for operator==, set
#include "vatensor//varray.h"         // for VArray, strides_type, axes_type
#include "xtensor/xlayout.hpp"        // for layout_type
#include "xtensor/xmanipulation.hpp"  // for full, transpose, flip, moveaxis
#include "xtensor/xstrided_view.hpp"  // for reshape_view

using namespace va;

std::shared_ptr<VArray> va::transpose(const VArray& varray, strides_type permutation) {
    return map([permutation](auto& array) {
        return xt::transpose(
            array,
            permutation,
            xt::check_policy::full{}
        );
    }, varray);
}

std::shared_ptr<VArray> va::reshape(const VArray& varray, strides_type new_shape) {
    return map([new_shape](auto& array) {
        auto new_shape_ = new_shape;
        return xt::reshape_view(array, new_shape_);
    }, varray);
}

std::shared_ptr<VArray> va::swapaxes(const VArray& varray, std::ptrdiff_t a, std::ptrdiff_t b) {
    return map([a, b](auto& array) {
        return xt::swapaxes(array, a, b);
    }, varray);
}

std::shared_ptr<VArray> va::moveaxis(const VArray& varray, std::ptrdiff_t src, std::ptrdiff_t dst) {
    return map([src, dst](auto& array) {
        return xt::moveaxis(array, src, dst);
    }, varray);
}

std::shared_ptr<VArray> va::flip(const VArray& varray, std::size_t axis) {
    return map([axis](auto& array) {
        return xt::flip(array, axis);
    }, varray);
}

template <typename T, typename I>
void move_indices_to_back(T& vec, const I& indices) {
    using ValueType = typename T::value_type;
    std::set<ValueType> indexSet(indices.begin(), indices.end());

    std::stable_partition(vec.begin(), vec.end(), [&indexSet](const ValueType& value) {
        // .contains is C++20
        return indexSet.find(value) == indexSet.end();
    });
}

std::shared_ptr<VArray> va::join_axes_into_last_dimension(const VArray &varray, axes_type axes) {
    const auto reduction_count = axes.size();

    if (reduction_count == 0) {
        return std::make_shared<VArray>(varray);
    }

    auto permutation = axes_type(varray.dimension());

    std::iota(permutation.begin(), permutation.end(), 0);
    move_indices_to_back(permutation, axes);

    return map([permutation, reduction_count](auto& carray) {
        auto transposed = xt::transpose(
            carray,
            permutation,
            xt::check_policy::full{}
        );
        shape_type new_shape = transposed.shape();
        auto reduction_begin = new_shape.end() - reduction_count;
        *reduction_begin = std::accumulate(reduction_begin, new_shape.end(), static_cast<std::size_t>(1), std::multiplies());
        new_shape.erase(reduction_begin + 1, new_shape.end());
        return xt::reshape_view(transposed, new_shape);
    }, varray);
}
