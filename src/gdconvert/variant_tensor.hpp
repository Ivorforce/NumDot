#ifndef VARIANT_TENSOR_HPP
#define VARIANT_TENSOR_HPP

#include "godot_cpp/variant/variant.hpp"
#include <xtensor/xadapt.hpp>
#include "vatensor/vcarray.hpp"

namespace numdot {
	// Base case
	template<typename T, std::size_t... Dims>
	struct ArrayAsTensor {
		using value_type = T;
		using shape = xt::fixed_shape<Dims...>;
		static constexpr std::size_t dimension = sizeof...(Dims);
	};

	// Recursive case
	template<typename T, std::size_t N, std::size_t... Dims>
	struct ArrayAsTensor<T[N], Dims...> : ArrayAsTensor<T, Dims..., N> {};

	template <typename T>
	static auto adapt_tensor(const T& t) {
		using Tensor = ArrayAsTensor<T>;
		return xt::adapt(reinterpret_cast<const typename Tensor::value_type*>(&t), typename Tensor::shape {});
	}

	template <typename T>
	static auto adapt_tensor(T& t) {
		using Tensor = ArrayAsTensor<T>;
		return xt::adapt(reinterpret_cast<typename Tensor::value_type*>(&t), typename Tensor::shape {});
	}

	template <typename V>
	struct VariantAsArray {};

	template <>
	struct VariantAsArray<godot::Vector2i> {
		static const auto& get(const godot::Vector2i& v) { return v.coord; }
		static auto& get(godot::Vector2i& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<godot::Vector3i> {
		static const auto& get(const godot::Vector3i& v) { return v.coord; }
		static auto& get(godot::Vector3i& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<godot::Vector4i> {
		static const auto& get(const godot::Vector4i& v) { return v.coord; }
		static auto& get(godot::Vector4i& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<godot::Vector2> {
		static const auto& get(const godot::Vector2& v) { return v.coord; }
		static auto& get(godot::Vector2& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<godot::Vector3> {
		static const auto& get(const godot::Vector3& v) { return v.coord; }
		static auto& get(godot::Vector3& v) { return v.coord; }
	};

	template <>
	struct VariantAsArray<godot::Vector4> {
		static const auto& get(const godot::Vector4& v) { return v.components; }
		static auto& get(godot::Vector4& v) { return v.components; }
	};

	template <>
	struct VariantAsArray<godot::Color> {
		static const auto& get(const godot::Color& v) { return v.components; }
		static auto& get(godot::Color& v) { return v.components; }
	};

	template <>
	struct VariantAsArray<godot::Quaternion> {
		static const auto& get(const godot::Quaternion& v) { return v.components; }
		static auto& get(godot::Quaternion& v) { return v.components; }
	};

	template <>
	struct VariantAsArray<godot::Plane> {
		static const auto& get(const godot::Plane& v) { return reinterpret_cast<const real_t(&)[4]>(v); }
		static auto& get(godot::Plane& v) { return reinterpret_cast<real_t(&)[4]>(v); }
	};

	template <>
	struct VariantAsArray<godot::Basis> {
		static const auto& get(const godot::Basis& v) { return reinterpret_cast<const real_t(&)[3][3]>(v); }
		static auto& get(godot::Basis& v) { return reinterpret_cast<real_t(&)[3][3]>(v); }
	};

	template <>
	struct VariantAsArray<godot::Projection> {
		static const auto& get(const godot::Projection& v) { return reinterpret_cast<const real_t(&)[4][4]>(v); }
		static auto& get(godot::Projection& v) { return reinterpret_cast<real_t(&)[4][4]>(v); }
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

		using Tensor = ArrayAsTensor<std::remove_reference_t<decltype(array)>>;
		constexpr auto shape_array = typename Tensor::shape {};

		const va::shape_type& shape = va::shape(data);

		ERR_FAIL_COND_V_MSG(shape.size() != shape_array.size(), {}, "shape is incompatible");
		ERR_FAIL_COND_V_MSG(!std::equal(shape.begin(), shape.end(), shape_array.begin()), {}, "shape is incompatible");

		try {
			va::util::fill_c_array_flat(reinterpret_cast<typename Tensor::value_type*>(&array), data);
		}
		catch (std::runtime_error& error) {
			ERR_FAIL_V_MSG({}, error.what());
		}

		return tensor;
	}
}

#endif //VARIANT_TENSOR_HPP
