#ifndef TRIGONOMETRY_H
#define TRIGONOMETRY_H

#include "varray.hpp"

namespace va {
	void sin(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void cos(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void tan(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);

	void asin(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void acos(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void atan(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void atan2(VStoreAllocator& allocator, const VArrayTarget& target, const VData& x1, const VData& x2);

	void sinh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void cosh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void tanh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);

	void asinh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void acosh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void atanh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);

	void angle(VStoreAllocator& allocator, const VArrayTarget& target, const std::shared_ptr<VArray>& array);
}

#endif //TRIGONOMETRY_H
