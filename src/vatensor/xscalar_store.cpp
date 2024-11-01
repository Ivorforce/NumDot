#include "xscalar_store.hpp"

va::VWrite va::store::VScalarStore::make_write(const VRead& read) {
	return std::visit(
	[](const auto& read) -> VWrite {
			using V = typename std::decay_t<decltype(read)>::value_type;
			// TODO should be layout_type::any
			// We can just const_cast the pointer; it doesn't matter for scalars.
			return make_compute<V*>(const_cast<V*>(read.data()), shape_type{}, strides_type{}, xt::layout_type::column_major);
		}, read
	);
}

va::VWrite va::store::VScalarStoreNonwrite::make_write(const VRead& read) {
	throw std::runtime_error("attempted to write to read-only storage");
}
