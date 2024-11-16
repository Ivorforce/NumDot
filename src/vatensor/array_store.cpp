#include "array_store.hpp"

void* va::store::VCharPtrStore::data() {
	return ptr;
}

va::DType va::store::VCharPtrStore::dtype() {
	return ptr_dtype;
}

std::size_t va::store::VCharPtrStore::size() {
	return n_bytes / va::size_of_dtype_in_bytes(ptr_dtype);
}

va::store::VCharPtrStore::~VCharPtrStore() {
	if (ptr != nullptr) {
		std::allocator<char>{}.deallocate(ptr, n_bytes);
	}
}
