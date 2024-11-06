#ifndef COMPARISON_H
#define COMPARISON_H

#include "varray.hpp"

namespace va {
	void equal_to(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void not_equal_to(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void greater(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void greater_equal(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void less(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void less_equal(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
}

#endif //COMPARISON_H
