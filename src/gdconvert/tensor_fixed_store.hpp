#ifndef TENSOR_FIXED_STORE_HPP
#define TENSOR_FIXED_STORE_HPP

#include <godot_cpp/core/defs.hpp>
#include <xtensor/xtensor_forward.hpp>
#include <vatensor/varray.hpp>

#include "variant_tensor.hpp"

namespace numdot {
	template <typename Type, typename FSH>
	class XTensorFixedStore : public va::VStore {
	public:
		using Tensor = xt::xtensor_fixed<Type, FSH>;

		Tensor tensor;
		explicit XTensorFixedStore(Tensor&& tensor) : tensor(std::forward<Tensor>(tensor)) {}

		void* data() override { return tensor.data(); }
		va::DType dtype() override { return va::dtype_of_type<typename Tensor::value_type>(); }
		size_t size() override { return tensor.size(); }
	};

	template <typename T>
	using VStoreVariant = XTensorFixedStore<
		std::remove_const_t<typename ArrayAsTensor<std::remove_reference_t<decltype(VariantAsArray<T>::get(std::declval<T>()))>>::value_type>,
		typename ArrayAsTensor<std::remove_reference_t<decltype(VariantAsArray<T>::get(std::declval<T>()))>>::shape
	>;

	template<typename Store>
	static std::shared_ptr<va::VArray> varray_from_tensor(typename Store::Tensor&& tensor) {
		using V = typename Store::Tensor::value_type;

		constexpr std::size_t dim = Store::Tensor::shape_type::size();

		auto shape = va::shape_type(dim);
		std::copy_n(tensor.shape().begin(), dim, shape.begin());

		auto strides = va::strides_type(dim);
		std::copy_n(tensor.strides().begin(), dim, strides.begin());

		auto store = std::make_shared<Store>(Store { std::forward<typename Store::Tensor>(tensor) });
		auto compute = va::make_compute<V*>(
			store->tensor.data() + store->tensor.data_offset(),  // Offset should be 0, but you know...
			shape,
			strides,
			store->tensor.layout()
		);

		return std::make_shared<va::VArray>(
			va::VArray {
				std::shared_ptr<va::VStore>(store),
				compute,
				static_cast<std::ptrdiff_t>(store->tensor.data_offset())
			}
		);
	}

	template<typename T>
	static std::shared_ptr<va::VArray> varray_from_tensor(const T& tensor) {
		using Array = VariantAsArray<T>;
		using StoreTensor = typename VStoreVariant<T>::Tensor;

		StoreTensor store_tensor;
		std::copy_n(reinterpret_cast<const typename StoreTensor::value_type*>(Array::get(tensor)), store_tensor.size(), store_tensor.linear_begin());

		return numdot::varray_from_tensor<VStoreVariant<T>>(
			std::move(store_tensor)
		);
	}
}

#endif //TENSOR_FIXED_STORE_HPP
