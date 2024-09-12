#include "allocate.h"

#include <memory>                                       // for make_shared
#include <variant>                                      // for visit
#include <vector>                                       // for vector
#include "vatensor/varray.h"                            // for dtype_to_variant
#include "xtensor/xarray.hpp"                           // for xarray_container
#include "xtensor/xbuilder.hpp"                         // for empty
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xstorage.hpp"                         // for uvector
#include "xtensor/xtensor_forward.hpp"                  // for xarray


va::VArray va::full(DType dtype, DTypeVariant fill_value, shape_type shape) {
    return std::visit([shape](auto t, auto fill_value) {
        using T = decltype(t);
        auto store = std::make_shared<xt::xarray<T>>(xt::xarray<T>::from_shape(shape));
        store->fill(static_cast<T>(fill_value));
        return from_store(store);
    }, dtype_to_variant(dtype), fill_value);
}

va::VArray va::empty(DType dtype, shape_type shape) {
    return std::visit([shape](auto t) {
        using T = decltype(t);
        auto store = std::make_shared<xt::xarray<T>>(xt::empty<T>(shape));
        return from_store(store);
    }, dtype_to_variant(dtype));
}

va::VArray va::copy_as_dtype(const VArray &other, DType dtype) {
    return std::visit([](auto t, auto carray) -> VArray {
        using T = decltype(t);
        return from_store(std::make_shared<xt::xarray<T>>(carray));
    }, dtype_to_variant(dtype), other.to_compute_variant());
}
