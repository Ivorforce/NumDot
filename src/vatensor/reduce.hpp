#ifndef REDUCE_H
#define REDUCE_H

#include "auto_defines.hpp"
#include "varray.hpp"

namespace va {
	VScalar sum(const VArray& array);
	void sum(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar prod(const VArray& array);
	void prod(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar mean(const VArray& array);
	void mean(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar median(const VArray& array);
	void median(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar var(const VArray& array);
	void var(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar std(const VArray& array);
	void std(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar max(const VArray& array);
	void max(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar min(const VArray& array);
	void min(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar norm_l0(const VArray& array);
	void norm_l0(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar norm_l1(const VArray& array);
	void norm_l1(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar norm_l2(const VArray& array);
	void norm_l2(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar norm_linf(const VArray& array);
	void norm_linf(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar count_nonzero(VStoreAllocator& allocator, const VArray& array);
	void count_nonzero(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	bool all(const VArray& array);
	void all(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	bool any(const VArray& array);
	void any(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const axes_type& axes);

	VScalar reduce_dot(const VArray& a, const VArray& b);
	void reduce_dot(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b, const axes_type& axes);
}

#endif //REDUCE_H
