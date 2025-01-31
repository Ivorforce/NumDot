#ifndef COMPARISON_H
#define COMPARISON_H

#include "varray.hpp"

namespace va {
	void array_equal(::va::VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	bool all_close(const VData& a, const VData& b, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
}

#endif //COMPARISON_H
