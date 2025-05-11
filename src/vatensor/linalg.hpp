#ifndef LINALG_H
#define LINALG_H

#include "varray.hpp"

namespace va {
	void dot(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void cross(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, std::ptrdiff_t axisa=-1, std::ptrdiff_t axisb=-1, std::ptrdiff_t axisc=-1);
	void matmul(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);

	void outer(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void inner(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
}

#endif //LINALG_H
