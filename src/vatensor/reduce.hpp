#ifndef REDUCE_H
#define REDUCE_H

#include "varray.hpp"

namespace va {
	VScalar sum(const VData& array);
	void sum(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar prod(const VData& array);
	void prod(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar mean(const VData& array);
	void mean(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar median(const VData& array);
	void median(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar variance(const VData& array);
	void variance(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar standard_deviation(const VData& array);
	void standard_deviation(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar max(const VData& array);
	void max(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar min(const VData& array);
	void min(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar norm_l0(const VData& array);
	void norm_l0(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar norm_l1(const VData& array);
	void norm_l1(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar norm_l2(const VData& array);
	void norm_l2(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar norm_linf(const VData& array);
	void norm_linf(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar count_nonzero(VStoreAllocator& allocator, const VData& array);
	void count_nonzero(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	bool all(const VData& array);
	void all(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	bool any(const VData& array);
	void any(VStoreAllocator& allocator, VArrayTarget target, const VData& array, const axes_type& axes);

	VScalar reduce_dot(const VData& a, const VData& b);
	void reduce_dot(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b, const axes_type& axes);

	void trace(VStoreAllocator& allocator, VArrayTarget target, const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2);
	VScalar trace_to_scalar(const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2);
}

#endif //REDUCE_H
