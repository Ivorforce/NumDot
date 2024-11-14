#ifndef VMATH_H
#define VMATH_H

#include "varray.hpp"

namespace va {
	void positive(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a);
	void negative(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a);

	void add(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void subtract(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void multiply(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void divide(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void remainder(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void pow(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);

	void minimum(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void maximum(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void clip(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& lo, const VData& hi);

	void sign(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void abs(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void square(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void sqrt(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void exp(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void log(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);

	void rad2deg(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);
	void deg2rad(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);

	void conjugate(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array);

	void a0xb1_minus_a1xb0(va::VStoreAllocator& allocator, const va::VArrayTarget& target, const va::VData& a, const va::VData& b, std::ptrdiff_t i0, std::ptrdiff_t i1);
}

#endif //MATH_H
