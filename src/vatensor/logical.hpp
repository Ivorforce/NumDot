#ifndef LOGICAL_H
#define LOGICAL_H

#include "varray.hpp"

namespace va {
	void logical_and(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void logical_or(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void logical_xor(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void logical_not(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a);
}

#endif //LOGICAL_H
