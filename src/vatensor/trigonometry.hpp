#ifndef TRIGONOMETRY_H
#define TRIGONOMETRY_H

#include "varray.hpp"

namespace va {
	void sin(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void cos(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void tan(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);

	void asin(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void acos(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void atan(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void atan2(VStoreAllocator& allocator, VArrayTarget target, const VArray& x1, const VArray& x2);

	void sinh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void cosh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void tanh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);

	void asinh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void acosh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void atanh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);

	void angle(VStoreAllocator& allocator, VArrayTarget target, const std::shared_ptr<VArray>& array);
}

#endif //TRIGONOMETRY_H
