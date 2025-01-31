#ifndef REDUCE_H
#define REDUCE_H

#include "varray.hpp"

namespace va {
	VScalar median(const VData& array);
	void median(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	VScalar max(const VData& array);
	void max(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	VScalar min(const VData& array);
	void min(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	VScalar norm_l0(const VData& array);
	void norm_l0(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	VScalar norm_l1(const VData& array);
	void norm_l1(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	VScalar norm_l2(const VData& array);
	void norm_l2(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	VScalar norm_linf(const VData& array);
	void norm_linf(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	VScalar count_nonzero(VStoreAllocator& allocator, const VData& array);
	void count_nonzero(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	bool all(const VData& array);
	void all(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	bool any(const VData& array);
	void any(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type& axes);

	VScalar reduce_dot(const VData& a, const VData& b);
	void reduce_dot(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, const axes_type& axes);

	void trace(VStoreAllocator& allocator, const VArrayTarget& target, const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2);
	VScalar trace_to_scalar(const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2);
}

#endif //REDUCE_H
