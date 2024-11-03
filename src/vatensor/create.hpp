#ifndef VATENSOR_FILL_H
#define VATENSOR_FILL_H

#include "auto_defines.hpp"
#include "varray.hpp"

namespace va {
	std::shared_ptr<VArray> full(VStoreAllocator& allocator, VScalar fill_value, const shape_type& shape);
	std::shared_ptr<VArray> empty(VStoreAllocator& allocator, DType dtype, const shape_type& shape);
	std::shared_ptr<VArray> eye(VStoreAllocator& allocator, DType dtype, const shape_type& shape, int k);

	std::shared_ptr<VArray> copy(VStoreAllocator& allocator, const VData& read);
	std::shared_ptr<VArray> copy_as_dtype(VStoreAllocator& allocator, const VData& other, DType dtype);

	std::shared_ptr<VArray> tile(VStoreAllocator& allocator, const VArray& array, const shape_type& reps, bool inner);
}

#endif //VATENSOR_FILL_H
