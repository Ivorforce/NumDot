#ifndef ROUND_H
#define ROUND_H

#include "varray.hpp"

namespace va {
	void ceil(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void floor(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void trunc(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void round(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void nearbyint(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
}

#endif //ROUND_H
