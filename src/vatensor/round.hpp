#ifndef ROUND_H
#define ROUND_H

#include "varray.hpp"

namespace va {
	void ceil(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void floor(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void trunc(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void round(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void nearbyint(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
}

#endif //ROUND_H
