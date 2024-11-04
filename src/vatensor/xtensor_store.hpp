#ifndef VSTORE_HPP
#define VSTORE_HPP

#include <memory>
#include <cmath>                           // for double_t, float_t
#include <complex>
#include <cstddef>                         // for size_t
#include <cstdint>                         // for int16_t, int32_t, int64_t

#include "varray.hpp"

namespace va::store {
	template<typename T>
	using tensor_case = xt::xtensor<T, 1, xt::layout_type::row_major>;

	using XArrayStoreVariant = std::variant<
		tensor_case<bool>,
		tensor_case<float_t>,
		tensor_case<double_t>,
		tensor_case<std::complex<float_t>>,
		tensor_case<std::complex<double_t>>,
		tensor_case<int8_t>,
		tensor_case<int16_t>,
		tensor_case<int32_t>,
		tensor_case<int64_t>,
		tensor_case<uint8_t>,
		tensor_case<uint16_t>,
		tensor_case<uint32_t>,
		tensor_case<uint64_t>
	>;

	class XArrayStore : public VStore {
		public:
			XArrayStoreVariant array;
			explicit XArrayStore(XArrayStoreVariant&& array) : array(std::forward<XArrayStoreVariant>(array)) {}

			void* data() override;
			DType dtype() override;
			size_t size() override;
	};

	class XArrayStoreAllocator: public VStoreAllocator {
		std::shared_ptr<VStore> allocate(DType dtype, std::size_t count) override;
	};

	static XArrayStoreAllocator default_allocator = {};

	// For deducted V, from xexpressions
	template<typename T>
	static std::shared_ptr<VStore> make_store(std::size_t count) {
		return std::shared_ptr<VStore>(std::make_shared<XArrayStore>(XArrayStore {
			XArrayStoreVariant { tensor_case<T>(typename tensor_case<T>::shape_type { count }) }
		}));
	}
}

#endif //VSTORE_HPP
