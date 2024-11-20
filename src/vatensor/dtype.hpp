#ifndef VATENSOR_DTYPE_HPP
#define VATENSOR_DTYPE_HPP

#include "varray.hpp"

namespace va {
	static constexpr bool is_any_dtype(const DType dtype) noexcept {
		return static_cast<size_t>(dtype) <= DTypeMax;
	}

	static constexpr bool is_normal_dtype(const DType dtype) noexcept {
		return static_cast<size_t>(dtype) <= DTypeMax;
	}

	template<typename Variant, std::size_t... Is>
	constexpr auto make_variant_array(std::index_sequence<Is...>) {
		return std::array<Variant, sizeof...(Is)> { Variant{std::in_place_index_t<Is>{}}... };
	}

	constexpr auto _variant_table = make_variant_array<VScalar>(
		std::make_index_sequence<std::variant_size_v<VScalar>>{}
	);

	static constexpr VScalar dtype_to_variant_unchecked(const DType dtype) noexcept {
		return _variant_table[static_cast<size_t>(dtype)];
	}

	static VScalar dtype_to_variant(const DType dtype) {
		static_assert(std::is_trivially_copyable_v<VScalar>, "VScalar is not trivially copyable");

		if (!is_normal_dtype(dtype)) {
			throw std::runtime_error("Invalid dtype.");
		}

		return dtype_to_variant_unchecked(dtype);
	}

	// Helper to retrieve the size of each type in the variant
	template<typename... Ts>
	constexpr auto make_size_array(std::variant<Ts...>) {
		return std::array<std::size_t, sizeof...(Ts)> {sizeof(Ts)...};
	}

	constexpr auto _size_table = make_size_array(VScalar{});

	static constexpr std::size_t size_of_dtype_in_bytes_unchecked(const DType dtype) noexcept {
		return _size_table[static_cast<size_t>(dtype)];
	}

	static std::size_t size_of_dtype_in_bytes(const DType dtype) {
		if (!is_normal_dtype(dtype)) {
			throw std::runtime_error("Invalid dtype.");
		}
		return size_of_dtype_in_bytes_unchecked(dtype);
	}

	// Function to create a lookup table for common DType
	constexpr auto create_common_type_table() {
		constexpr auto row_count = DType::DTypeMax + 1;
		std::array<std::array<DType, row_count>, row_count> table{};

		for (int i = 0; i < DType::DTypeMax; ++i) {
			for (int j = 0; j < DType::DTypeMax; ++j) {
				table[i][j] = std::visit(
					[](auto a, auto b) {
						using CommonType = std::common_type_t<decltype(a), decltype(b)>;
						return dtype_of_type<CommonType>();
					},
					dtype_to_variant_unchecked(DType(i)),
					dtype_to_variant_unchecked(DType(j))
				);
			}
		}
		for (int i = 0; i <= DType::DTypeMax; ++i) {
			table[DType::DTypeMax][i] = static_cast<DType>(i);
			table[i][DType::DTypeMax] = static_cast<DType>(i);
		}

		return table;
	}

	constexpr auto _common_type_table = create_common_type_table();

	static constexpr DType dtype_common_type_unchecked(const DType a, const DType b) noexcept {
		return _common_type_table[static_cast<size_t>(a)][static_cast<size_t>(b)];
	}

	static DType dtype_common_type(const DType a, const DType b) {
		if (!is_any_dtype(a) || !is_any_dtype(b)) {
			throw std::runtime_error("Invalid dtype.");
		}
		return dtype_common_type_unchecked(a, b);
	}

	static DType complex_dtype_value_type(const DType dtype) {
		return std::visit([](auto t) -> DType {
			if constexpr (xtl::is_complex<decltype(t)>::value) {
				return dtype_of_type<typename decltype(t)::value_type>();
			}
			else {
				throw std::runtime_error("DType must be complex");
			}
		}, dtype_to_variant(dtype));
	}
}

#endif //VATENSOR_DTYPE_HPP
