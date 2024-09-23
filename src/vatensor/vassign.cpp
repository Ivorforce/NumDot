#include "vassign.h"
#include "xtensor/xassign.hpp"

using namespace va;

// computed_assign on containers doesn't assign data, it tries to assign to the whole container.
// This is basically view_semantic's computed_assign.
template <typename T, typename E>
inline void broadcasting_assign(xt::xexpression<T>& t, const xt::xexpression<E>& e) {
    xt::assert_compatible_shape(t, e);
    xt::assign_data(t, e, xt::detail::get_rhs_triviality(e.derived_cast()));
}

void va::assign(ComputeVariant& array, const ComputeVariant& value) {
    std::visit([](auto& carray, const auto& cvalue) {
        using T = typename std::decay_t<decltype(carray)>::value_type;
        // Cast first to reduce number of combinations down the line.
        broadcasting_assign(carray, xt::cast<T>(cvalue));
    }, array, value);
}

void va::assign(ComputeVariant& array, const ArrayVariant& value) {
    std::visit([](auto& carray, const auto& cvalue) {
        using T = typename std::decay_t<decltype(carray)>::value_type;
        // Cast first to reduce number of combinations down the line.
        broadcasting_assign(carray, xt::cast<T>(cvalue));
    }, array, value);
}

void va::assign(ComputeVariant& array, VConstant value) {
    std::visit([](auto& carray, const auto cvalue) {
        // Cast first to reduce number of combinations down the line.
        using T = typename std::decay_t<decltype(carray)>::value_type;
        carray.fill(static_cast<T>(cvalue));
    }, array, value);
}
