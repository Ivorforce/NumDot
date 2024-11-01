#include "xarray_store.hpp"

using namespace va;

VWrite store::XArrayStore::make_write(const VRead& read) {
	return std::visit(
		[](const auto& read) -> VWrite {
			using V = typename std::decay_t<decltype(read)>::value_type;
			return make_compute<V*>(const_cast<V*>(read.data()), read.shape(), read.strides(), read.layout());
		},  read
	);
}
