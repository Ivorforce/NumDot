#ifndef XSCALAR_STORE_HPP
#define XSCALAR_STORE_HPP

#include "varray.hpp"

namespace va::store {
	class VScalarStore : public VStore {
	public:
		VScalar scalar;
		explicit VScalarStore(VScalar scalar) : scalar(scalar) {}

		VWrite make_write(const VRead& read) override;
	};

	template<typename V>
	static std::shared_ptr<VArray> from_scalar(const V value) {
		auto store = std::make_shared<VScalarStore>(VScalarStore { value });

		return std::make_shared<VArray>(VArray {
			std::shared_ptr<VStore>(store),
			make_compute<const V*>(
				// Point to the store's value.
				static_cast<const V*>(&std::get<V>(store->scalar)),
				shape_type{},
				strides_type{},
				xt::layout_type::row_major  // TODO Should be any
			)
		});
	}
}
#endif //XSCALAR_STORE_HPP
