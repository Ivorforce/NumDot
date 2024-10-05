#include "vassign.hpp"

#include <type_traits>                                  // for decay_t
#include <variant>                                      // for visit
#include "varray.hpp"                            // for VWrite, VScalar
#include <xtensor/xmasked_view.hpp>

using namespace va;

void va::assign(VWrite& array, const VRead& value) {
	std::visit(
		[](auto& carray, const auto& cvalue) {
			broadcasting_assign(carray, cvalue);
		}, array, value
	);
}

void va::assign_nonoverlapping(VWrite& array, const ArrayVariant& value) {
	std::visit(
		[](auto& carray, const auto& cvalue) {
			broadcasting_assign(carray, cvalue);
		}, array, value
	);
}

void va::assign(VWrite& array, VScalar value) {
	std::visit(
		[](auto& carray, const auto cvalue) {
			using V = typename std::decay_t<decltype(carray)>::value_type;
			carray.fill(static_cast<V>(cvalue));
		}, array, value
	);
}

void va::assign(VArrayTarget target, VScalar value) {
	std::visit(
		[value](auto target) {
			if constexpr (std::is_same_v<decltype(target), VWrite*>) {
				va::assign(*target, value);
			}
			else {
				*target = from_scalar_variant(value);
			}
		}, target
	);
}

std::shared_ptr<VArray> va::get_at_mask(const VRead& varray, const VRead& mask) {
	return std::visit(
		[](const auto& array, const auto& mask) -> std::shared_ptr<VArray> {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTMask = typename std::decay_t<decltype(mask)>::value_type;

			if constexpr (!std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be boolean dtype");
			}
			else {
				// Masked views don't offer this functionality automatically.
				const size_type array_size = xt::sum(mask)();
				auto result = make_store<VTArray>(xt::empty<VTArray>({ array_size }));
				const auto masked_view = xt::masked_view(array, mask);

				auto iter_result = result->begin();
				for (auto masked_value : masked_view) {
					if (masked_value.visible()) {
						*iter_result = masked_value.value();
						++iter_result;
					}
				}

				return from_store(result);
			}
		}, varray, mask
	);
}

void va::set_at_mask(VWrite& varray, VRead& mask, VRead& value) {
	return std::visit(
		// Mask can't be const because of masked_view iterator.
		[](auto& array, auto& mask, const auto& value) {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTMask = typename std::decay_t<decltype(mask)>::value_type;

			if constexpr (!std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be boolean dtype");
			}
			else {
				auto masked_view = xt::masked_view(array, mask);
				if (value.dimension() == 0) {
					// Simple fill, masked_view supports this.
					masked_view = *value.data();
					return;
				}

				// Masked views don't offer array fill functionality automatically.
				const size_type array_size = xt::sum(mask)();
				if (value.shape() != shape_type { array_size })
					throw std::runtime_error("mask must be single value or match the mask sum");

				const auto stride = value.strides()[0];
				auto iter_value = array.begin();
				for (auto masked_value : masked_view) {
					if (masked_value.visible()) {
						*iter_value = masked_value.value();
						iter_value += stride;
					}
				}
			}
		}, varray, mask, value
	);
}
