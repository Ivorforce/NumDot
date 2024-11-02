#ifndef PACKED_ARRAY_STORE_HPP
#define PACKED_ARRAY_STORE_HPP

#include <godot_cpp/core/defs.hpp>
#include <xtensor/xtensor_forward.hpp>
#include <vatensor/varray.hpp>

namespace numdot {
	template <typename Type, typename FSH>
	class XTensorFixedStore : public va::VStore {
	public:
		using Tensor = xt::xtensor_fixed<Type, FSH>;

		Tensor tensor;
		explicit XTensorFixedStore(Tensor&& tensor) : tensor(std::forward<Tensor>(tensor)) {}
	};

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
				compute
			}
		);
	}

	using VStoreVector2i = XTensorFixedStore<int32_t, xt::xshape<2>>;
	using VStoreVector3i = XTensorFixedStore<int32_t, xt::xshape<3>>;
	using VStoreVector4i = XTensorFixedStore<int32_t, xt::xshape<4>>;
	using VStoreVector2 = XTensorFixedStore<real_t, xt::xshape<2>>;
	using VStoreVector3 = XTensorFixedStore<real_t, xt::xshape<3>>;
	using VStoreVector4 = XTensorFixedStore<real_t, xt::xshape<4>>;
	using VStoreColor = XTensorFixedStore<float_t, xt::xshape<4>>;
}

#endif //PACKED_ARRAY_STORE_HPP
