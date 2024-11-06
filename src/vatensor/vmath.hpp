#ifndef VMATH_H
#define VMATH_H

#include "varray.hpp"

namespace va {
	void positive(VStoreAllocator& allocator, VArrayTarget target, const VArray& a);
	void negative(VStoreAllocator& allocator, VArrayTarget target, const VArray& a);

	void add(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void subtract(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void multiply(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void divide(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void remainder(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void pow(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);

	void minimum(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void maximum(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b);
	void clip(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& lo, const VArray& hi);

	void sign(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void abs(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void square(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void sqrt(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void exp(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void log(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);

	void rad2deg(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
	void deg2rad(VStoreAllocator& allocator, VArrayTarget target, const VArray& array);
}

#endif //MATH_H
