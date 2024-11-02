#include "xscalar_store.hpp"

void va::store::VScalarStoreNonwrite::prepare_write(VData& data) {
	throw std::runtime_error("attempted to write to read-only storage");
}
