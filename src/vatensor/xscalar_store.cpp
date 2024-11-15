#include "xscalar_store.hpp"

void* va::store::VScalarStore::data() {
	return std::visit([](auto& value) -> void* {
		return &value;
	}, scalar);
}

va::DType va::store::VScalarStore::dtype() {
	return va::dtype(scalar);
}

void va::store::VScalarStoreNonwrite::prepare_write(VData& data, std::ptrdiff_t data_offset) {
	throw std::runtime_error("attempted to write to read-only storage");
}

std::shared_ptr<va::VArray> va::store::from_scalar_variant(VScalar scalar) {
	return std::visit(
		[](auto cscalar) {
			return va::store::from_scalar(cscalar);
		}, scalar
	);
}
