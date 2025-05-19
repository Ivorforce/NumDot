#ifndef XSCALAR_STORE_HPP
#define XSCALAR_STORE_HPP

#include "varray.hpp"

namespace va::store {
	class VScalarStore : public VStore {
	public:
		VScalar scalar;

		explicit VScalarStore(VScalar scalar) : scalar(scalar) {}

		void* data() override;
		DType dtype() override;
		std::size_t size() override { return 1; }
	};

	class VScalarStoreNonwrite : public VScalarStore {
	public:
		explicit VScalarStoreNonwrite(VScalar scalar) : VScalarStore(scalar) {}

		void prepare_write(VData& data, std::ptrdiff_t data_offset) override;
	};

	template <typename V>
	std::shared_ptr<VArray> from_scalar(const V value) {
		auto store = std::make_shared<VScalarStore>(VScalarStore { value });

		return std::make_shared<VArray>(VArray {
			std::shared_ptr<VStore>(store),
			make_compute<V*>(
				// Point to the store's value.
				static_cast<V*>(&std::get<V>(store->scalar)),
				shape_type{},
				strides_type{},
				xt::layout_type::any
			),
			0
		});
	}

	template <typename V>
	std::shared_ptr<VArray> full_dummy_like(const V value, const VData& read) {
		auto store = std::make_shared<VScalarStoreNonwrite>(VScalarStoreNonwrite { value });

		strides_type strides(va::dimension(read));
		std::fill(strides.begin(), strides.end(), 0);

		return std::make_shared<VArray>(VArray {
			std::shared_ptr<VStore>(store),
			make_compute<V*>(
				// Point to the store's value.
				static_cast<V*>(&std::get<V>(store->scalar)),
				va::shape(read),
				strides,
				// Because strides are fake
				xt::layout_type::dynamic
			),
			0
		});
	}

	std::shared_ptr<VArray> from_scalar_variant(VScalar scalar);
}
#endif //XSCALAR_STORE_HPP
