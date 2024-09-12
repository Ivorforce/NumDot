#include "allocate.h"

#include <memory>                       // for make_shared
#include <utility>                      // for move
#include <variant>                      // for visit
#include "vatensor/varray.h"            // for VArray, shape_type, DType
#include "xtensor/xbuilder.hpp"         // for empty
#include "xtensor/xlayout.hpp"          // for layout_type
#include "xtensor/xoperation.hpp"       // for cast
#include "xtensor/xtensor_forward.hpp"  // for xarray

using namespace va;

VArray empty(VConstant type, shape_type shape) {
    return std::visit([shape](auto t) {
        using T = decltype(t);
        auto store = std::make_shared<xt::xarray<T>>(xt::empty<T>(shape));
        return from_store(store);
    }, type);
}

VArray va::full(const VConstant fill_value, shape_type shape) {
    // Could do this in one go but that would generate more code. This is fast enough.
    VArray va = ::empty(fill_value, std::move(shape));
    va.fill(fill_value);
    return va;
}

VArray va::empty(DType dtype, shape_type shape) {
    return ::empty(dtype, std::move(shape));
}

VArray va::copy_as_dtype(const VArray &other, DType dtype) {
    return std::visit([](auto t, auto carray) -> VArray {
        using T = decltype(t);
        // Cast first to reduce number of combinations down the line.
        return from_store(std::make_shared<xt::xarray<T>>(xt::cast<T>(carray)));
    }, dtype_to_variant(dtype), other.to_compute_variant());
}
