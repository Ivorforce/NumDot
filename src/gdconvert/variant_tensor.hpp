#ifndef VARIANT_TENSOR_HPP
#define VARIANT_TENSOR_HPP

#include <godot_cpp/variant/utility_functions.hpp>
#include <xtensor/xadapt.hpp>

namespace numdot {
	template <typename V>
	struct VariantAsArray {
	};

	template <>
	struct VariantAsArray<Vector2i> {
		static const auto& get(const Vector2i& v) { return v.coord; }
		static auto& get(Vector2i& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<Vector3i> {
		static const auto& get(const Vector3i& v) { return v.coord; }
		static auto& get(Vector3i& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<Vector4i> {
		static const auto& get(const Vector4i& v) { return v.coord; }
		static auto& get(Vector4i& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<Vector2> {
		static const auto& get(const Vector2& v) { return v.coord; }
		static auto& get(Vector2& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<Vector3> {
		static const auto& get(const Vector3& v) { return v.coord; }
		static auto& get(Vector3& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<Vector4> {
		static const auto& get(const Vector4& v) { return v.components; }
		static auto& get(Vector4& v) { return v.components; }
	};

	template <>
	struct VariantAsArray<Color> {
		static const auto& get(const Color& v) { return v.components; }
		static auto& get(Color& v) { return v.components; }
	};

	template <typename T>
	using array_value_type = std::remove_const_t<std::remove_reference_t<decltype(*std::declval<T>())>>;

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

	template <typename P>
	P to_variant_tensor(va::VData& data) {
		P tensor;
		auto& array = numdot::VariantAsArray<P>::get(tensor);
		constexpr auto array_size = sizeof(array) / sizeof(*array);

		const va::shape_type& shape = va::shape(data);
		ERR_FAIL_COND_V_MSG(shape.size() != 1, {}, "flatten the array before conversion");
		ERR_FAIL_COND_V_MSG(shape[0] != array_size, {}, "shape is incompatible");

		try {
			va::util::fill_c_array_flat(array, data);
		}
		catch (std::runtime_error& error) {
			ERR_FAIL_V_MSG({}, error.what());
		}

		return tensor;
	}
}

#endif //VARIANT_TENSOR_HPP
