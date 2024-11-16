#ifndef VARIANT_TENSOR_HPP
#define VARIANT_TENSOR_HPP

#include <xtensor/xadapt.hpp>

namespace numdot {
	template <typename P, typename E, std::size_t... axes>
	P to_packed(va::VData& data) {
		const va::shape_type& shape = va::shape(data);
		constexpr std::array<std::size_t, sizeof...(axes)> axes_array = {axes...};

		ERR_FAIL_COND_V_MSG(shape.size() != (sizeof...(axes) + 1), {}, "flatten the array before conversion");
		for (std::size_t i = 0; i < sizeof...(axes); ++i) {
			ERR_FAIL_COND_V_MSG(shape[i + 1] != axes_array[i], {}, "shape is incompatible");
		}

		P packed;
		packed.resize(static_cast<int64_t>(shape[0]));

		try {
			va::util::fill_c_array_flat(reinterpret_cast<E*>(packed.ptrw()), data);
		}
		catch (std::runtime_error& error) {
			ERR_FAIL_V_MSG({}, error.what());
		}

		return packed;
	}

	template <typename P, typename E, std::size_t... axes>
	P to_variant_tensor(va::VData& data) {
		const va::shape_type& shape = va::shape(data);
		constexpr std::array<std::size_t, sizeof...(axes)> axes_array = {axes...};

		ERR_FAIL_COND_V_MSG(shape.size() != sizeof...(axes), {}, "flatten the array before conversion");
		for (std::size_t i = 0; i < sizeof...(axes); ++i) {
			ERR_FAIL_COND_V_MSG(shape[i] != axes_array[i], {}, "shape is incompatible");
		}

		P tensor;

		try {
			va::util::fill_c_array_flat(reinterpret_cast<E*>(&tensor), data);
		}
		catch (std::runtime_error& error) {
			ERR_FAIL_V_MSG({}, error.what());
		}

		return tensor;
	}
}

#endif //VARIANT_TENSOR_HPP
