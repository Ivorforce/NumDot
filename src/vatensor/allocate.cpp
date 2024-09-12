#include "allocate.h"

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
