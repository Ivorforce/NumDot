#include "xscalar_store.hpp"

void va::store::VScalarStoreNonwrite::prepare_write(VData& data) {
	throw std::runtime_error("attempted to write to read-only storage");
}

std::shared_ptr<va::VArray> va::store::from_scalar_variant(VScalar scalar) {
	return std::visit(
		[](auto cscalar) {
			return va::store::from_scalar(cscalar);
		}, scalar
	);
}
