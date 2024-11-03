#ifndef ROUND_H
#define ROUND_H

#include "auto_defines.hpp"
#include "varray.hpp"

namespace va {
	void ceil(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void floor(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void trunc(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void round(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void nearbyint(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
}

#endif //ROUND_H
