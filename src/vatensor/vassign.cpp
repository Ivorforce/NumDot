#include "vassign.h"
#include "xtensor/xview.hpp"

using namespace va;

// TODO xt::view is needed because xarray_adaptor is container_semantic,
//  which is expected to resize on assignments.
// We should find a better solution...

void va::assign(ComputeVariant& array, const ComputeVariant& value) {
    std::visit([](auto& carray, const auto& cvalue) {
        using T = typename std::decay_t<decltype(carray)>::value_type;
        // Cast first to reduce number of combinations down the line.
        xt::view(carray, xt::all()).computed_assign(xt::cast<T>(cvalue));
    }, array, value);
}

void va::assign(ComputeVariant& array, const ArrayVariant& value) {
    std::visit([](auto& carray, const auto& cvalue) {
        using T = typename std::decay_t<decltype(carray)>::value_type;
        // Cast first to reduce number of combinations down the line.
        xt::view(carray, xt::all()).assign_xexpression(xt::cast<T>(cvalue));
    }, array, value);
}

void va::assign(ComputeVariant& array, VConstant value) {
    std::visit([](auto& carray, const auto cvalue) {
        // Cast first to reduce number of combinations down the line.
        using T = typename std::decay_t<decltype(carray)>::value_type;
        carray.fill(static_cast<T>(cvalue));
    }, array, value);
}
