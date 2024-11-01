#include "xscalar_store.hpp"

va::VWrite va::store::VScalarStore::make_write(const VRead& read) {
	return std::visit(
	[](const auto& read) -> VWrite {
			using V = typename std::decay_t<decltype(read)>::value_type;
			// You may expect scalars to always be dimension 0, but who knows how we'll be re-interpreted...
			return make_compute<V*>(const_cast<V*>(read.data()), read.shape(), read.strides(), read.layout());
		}, read
	);
}

va::VWrite va::store::VScalarStoreNonwrite::make_write(const VRead& read) {
	throw std::runtime_error("attempted to write to read-only storage");
}
