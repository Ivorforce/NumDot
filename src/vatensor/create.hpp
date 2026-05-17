#ifndef VATENSOR_FILL_H
#define VATENSOR_FILL_H

#include "varray.hpp"

namespace va {
	std::shared_ptr<VArray> full(VStoreAllocator& allocator, VScalar fill_value, const shape_type& shape);
	std::shared_ptr<VArray> empty(VStoreAllocator& allocator, DType dtype, const shape_type& shape);
	std::shared_ptr<VArray> eye(VStoreAllocator& allocator, DType dtype, const shape_type& shape, std::ptrdiff_t k);

	std::shared_ptr<VArray> copy(VStoreAllocator& allocator, const VData& read);
	std::shared_ptr<VArray> copy_as_dtype(VStoreAllocator& allocator, const VData& other, DType dtype);

	std::shared_ptr<VArray> linspace(VStoreAllocator& allocator, VScalar start, VScalar stop, std::size_t num, bool endpoint, DType dtype);
	std::shared_ptr<VArray> arange(VStoreAllocator& allocator, VScalar start, VScalar stop, VScalar step, DType dtype);

	std::shared_ptr<VArray> tile(VStoreAllocator& allocator, const VArray& array, const shape_type& reps, bool inner);

	std::shared_ptr<VArray> reshape(VStoreAllocator& allocator, const std::shared_ptr<VArray>& varray, strides_type new_shape);
	std::shared_ptr<VArray> flatten(VStoreAllocator& allocator, const std::shared_ptr<VArray>& varray);

	// Coordinate grids from N 1-D arrays. With "ij" indexing each output has
	// shape (len(x0), len(x1), ..., len(xN-1)); with "xy" indexing the first
	// two dims are swapped (and only that — higher dims are unchanged), so for
	// 2-D inputs the result is the (rows=y, cols=x) image-coordinate layout.
	// Outputs are independent materialized arrays (no shared storage).
	std::vector<std::shared_ptr<VArray>> meshgrid(VStoreAllocator& allocator, const std::vector<std::shared_ptr<VArray>>& inputs, bool xy_indexing);
}

#endif //VATENSOR_FILL_H
