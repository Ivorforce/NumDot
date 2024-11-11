#ifndef TRIGONOMETRY_H
#define TRIGONOMETRY_H

#include "varray.hpp"

namespace va {
	void sin(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void cos(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void tan(VStoreAllocator& allocator, VArrayTarget target, const VData& array);

	void asin(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void acos(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void atan(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void atan2(VStoreAllocator& allocator, VArrayTarget target, const VData& x1, const VData& x2);

	void sinh(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void cosh(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void tanh(VStoreAllocator& allocator, VArrayTarget target, const VData& array);

	void asinh(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void acosh(VStoreAllocator& allocator, VArrayTarget target, const VData& array);
	void atanh(VStoreAllocator& allocator, VArrayTarget target, const VData& array);

	void angle(VStoreAllocator& allocator, VArrayTarget target, const std::shared_ptr<VArray>& array);
}

#endif //TRIGONOMETRY_H
