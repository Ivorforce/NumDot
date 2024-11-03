#ifndef LOGICAL_H
#define LOGICAL_H

#include "auto_defines.hpp"
#include "varray.hpp"

namespace va {
	void logical_and(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void logical_or(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void logical_xor(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void logical_not(VStoreAllocator& allocator, VArrayTarget target, const VArray& a);
}

#endif //LOGICAL_H
