#ifndef VATENSOR_DTYPE_HPP
#define VATENSOR_DTYPE_HPP

#include <variant>
#include <exception>
#include <xtensor/core/xshape.hpp>
#include <xtensor/misc/xcomplex.hpp>

namespace va {
	// We should be using the same default types as xarray does, so we know for sure the ones we create /
	//  pass around are the ones we need in the end.
	using size_type = std::size_t;
	// Refer to xarray
	using shape_type = xt::dynamic_shape<size_type>;
	// Refer to xarray xcontainer_inner_types.
	using strides_type = xt::get_strides_t<shape_type>;
	using axes_type = strides_type;

	enum DType {
		Bool,
		Float32,
		Float64,
		Complex64,
		Complex128,
		Int8,
		Int16,
		Int32,
		Int64,
		UInt8,
		UInt16,
		UInt32,
		UInt64,
		DTypeMax
	};

	using VScalar = std::variant<
		bool,
		float_t,
		double_t,
		std::complex<float_t>,
		std::complex<double_t>,
		int8_t,
		int16_t,
		int32_t,
		int64_t,
		uint8_t,
		uint16_t,
		uint32_t,
		uint64_t
	>;

	constexpr static DType dtype(VScalar dtype) {
		return static_cast<DType>(dtype.index());
	}

	template <typename T>
	constexpr DType dtype_of_type() {
		return static_cast<DType>(VScalar(T()).index());
	}

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

	VScalar static_cast_scalar(VScalar v, DType dtype);

	template<typename VWrite, typename VRead>
	VWrite static_cast_scalar(VRead v) {
		if constexpr (std::is_same_v<VWrite, bool> && xtl::is_complex<VRead>::value) {
			// This helps mostly complex dtypes to booleanize
			return v != static_cast<decltype(v)>(0);
		}
		else if constexpr (!std::is_convertible_v<VRead, VWrite>) {
			throw std::runtime_error("Cannot promote in this way.");
		}
		else {
			return static_cast<VWrite>(v);
		}
	}

	template<typename VWrite>
	VWrite static_cast_scalar(const VScalar& v) {
		return std::visit([](const auto& v) -> VWrite {
			return static_cast_scalar<VWrite>(v);
		}, v);
	}
}

#endif //VATENSOR_DTYPE_HPP
