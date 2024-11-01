#include "xarray_store.hpp"

using namespace va;

// Need the store to request a write variant
template<typename V>
static compute_case<V*> make_vwrite(array_case<V>& array, const compute_case<const V*>& read) {
	auto offset = read.data() - array.data();
	return make_compute<V*>(array.data() + offset, read.shape(), read.strides(), read.layout());
}

VWrite store::XArrayStore::make_write(const VRead& read) {
	return std::visit(
		[](auto& array, const auto& read) -> VWrite {
			using VTStore = typename std::decay_t<decltype(array)>::value_type;
			using VTRead = typename std::decay_t<decltype(read)>::value_type;

			if constexpr (!std::is_same_v<VTStore, VTRead>) {
				throw std::runtime_error("unexpected data type discrepancy between store array and read");
			}
			else {
				return make_vwrite<VTStore>(array, read);
			}
		}, array, read
	);
}
