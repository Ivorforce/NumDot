#include "xtensor_store.hpp"

using namespace va;

void* store::XArrayStore::data() {
	return std::visit([](auto& array) -> void* {
		return array.data();
	}, array);
}

DType store::XArrayStore::dtype() {
	return std::visit([](auto& array) -> va::DType {
		using V = typename std::decay_t<decltype(array)>::value_type;
		return dtype_of_type<V>();
	}, array);
}

size_t store::XArrayStore::size() {
	return std::visit([](auto& array) -> std::size_t {
		return array.size();
	}, array);
}

std::shared_ptr<VStore> store::XArrayStoreAllocator::allocate(DType dtype, std::size_t count) {
	return std::visit([count](auto t) -> std::shared_ptr<VStore> {
		using T = std::decay_t<decltype(t)>;
		return make_store<T>(count);
	}, dtype_to_variant(dtype));
}
