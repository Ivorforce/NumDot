#include "vassign.h"
#include "xtensor/xassign.hpp"

using namespace va;

void va::assign(ComputeVariant& array, const ComputeVariant& value) {
    std::visit([](auto& carray, const auto& cvalue) {
        broadcasting_assign(carray, cvalue);
    }, array, value);
}

void va::assign_nonoverlapping(ComputeVariant& array, const ArrayVariant& value) {
    std::visit([](auto& carray, const auto& cvalue) {
        broadcasting_assign(carray, cvalue);
    }, array, value);
}

void va::assign(ComputeVariant& array, VConstant value) {
    std::visit([](auto& carray, const auto cvalue) {
        using V = typename std::decay_t<decltype(carray)>::value_type;
        carray.fill(static_cast<V>(cvalue));
    }, array, value);
}

void va::assign(VArrayTarget target, VConstant value) {
	std::visit([value](auto target) {
		if constexpr (std::is_same_v<decltype(target), ComputeVariant*>) {
			va::assign(*target, value);
		}
		else {
			*target = from_constant_variant(value);
		}
	}, target);
}
