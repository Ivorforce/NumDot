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
}

#endif //VMATH_H
