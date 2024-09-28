#include "vassign.h"

#include <type_traits>                                  // for decay_t
#include <variant>                                      // for visit
#include "vatensor/varray.h"                            // for VWrite, VScalar

using namespace va;

void va::assign(VWrite& array, const VRead& value) {
    std::visit([](auto& carray, const auto& cvalue) {
        broadcasting_assign(carray, cvalue);
    }, array, value);
}

void va::assign_nonoverlapping(VWrite& array, const ArrayVariant& value) {
    std::visit([](auto& carray, const auto& cvalue) {
        broadcasting_assign(carray, cvalue);
    }, array, value);
}

void va::assign(VWrite& array, VScalar value) {
    std::visit([](auto& carray, const auto cvalue) {
        using V = typename std::decay_t<decltype(carray)>::value_type;
        carray.fill(static_cast<V>(cvalue));
    }, array, value);
}

void va::assign(VArrayTarget target, VScalar value) {
	std::visit([value](auto target) {
		if constexpr (std::is_same_v<decltype(target), VWrite*>) {
			va::assign(*target, value);
		}
		else {
			*target = from_scalar_variant(value);
		}
	}, target);
}
