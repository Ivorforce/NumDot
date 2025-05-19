#ifndef VA_ARRAY_STORE_HPP
#define VA_ARRAY_STORE_HPP

#include "varray.hpp"
#include "xscalar_store.hpp"
#include "dtype.hpp"

namespace va::store {
	class VCharPtrStore : public VStore {
	public:
		char* ptr;
		std::size_t n_bytes;
		DType ptr_dtype;

		void* data() override;
		DType dtype() override;
		std::size_t size() override;

		VCharPtrStore(VCharPtrStore&& other) noexcept : ptr(other.ptr), n_bytes(other.n_bytes), ptr_dtype(other.ptr_dtype) {
			other.ptr = nullptr;
		}
		VCharPtrStore(char* ptr, std::size_t n_bytes, DType dtype) : ptr(ptr), n_bytes(n_bytes), ptr_dtype(dtype) {}
		~VCharPtrStore() override;
	};

	inline std::shared_ptr<VArray> from_store(VCharPtrStore&& store, const va::shape_type& shape, const va::strides_type& strides, const xt::layout_type layout) {
		auto compute = std::visit([&store, &shape, &strides, &layout](auto t) -> VData {
			using T = decltype(t);
			return make_compute<T*>(
				reinterpret_cast<T*>(store.ptr),
				shape,
				strides,
				layout
			);
		}, va::dtype_to_variant_unchecked(store.ptr_dtype));

		auto store_ = std::make_shared<VCharPtrStore>(std::forward<VCharPtrStore>(store));

		return std::make_shared<VArray>(VArray {
			std::shared_ptr<VStore>(store_),
			compute,
			0
		});
	}
}

#endif //VA_ARRAY_STORE_HPP
