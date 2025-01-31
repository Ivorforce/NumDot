#ifndef REDUCE_H
#define REDUCE_H

#include "varray.hpp"

namespace va {
	VScalar median(const VData& array);
	void median(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	void count_nonzero(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type* axes);

	void trace(VStoreAllocator& allocator, const VArrayTarget& target, const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2);
	VScalar trace_to_scalar(const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2);
}

#endif //REDUCE_H
