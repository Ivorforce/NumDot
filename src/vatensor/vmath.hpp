#ifndef VMATH_H
#define VMATH_H

#include "varray.hpp"
#include "vassign.hpp"
#include "vcall.hpp"

namespace va {
	inline void positive(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a) {
		va::assign(allocator, target, a);
	}

	void clip(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& lo, const VData& hi);

	void a0xb1_minus_a1xb0(va::VStoreAllocator& allocator, const va::VArrayTarget& target, const va::VData& a, const va::VData& b, std::ptrdiff_t i0, std::ptrdiff_t i1);
}

#endif //MATH_H
