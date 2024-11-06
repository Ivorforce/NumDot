#ifndef LINALG_H
#define LINALG_H

#include "varray.hpp"

namespace va {
	void dot(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void matmul(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
}

#endif //LINALG_H
