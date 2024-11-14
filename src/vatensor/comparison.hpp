#ifndef COMPARISON_H
#define COMPARISON_H

#include "varray.hpp"

namespace va {
	void equal_to(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void not_equal_to(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void greater(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void greater_equal(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void less(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void less_equal(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b);
	void is_close(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);

	void is_nan(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a);
	void is_finite(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a);
	void is_inf(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a);

	bool array_equal(const VData& a, const VData& b);
	bool all_close(const VData& a, const VData& b, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
}

#endif //COMPARISON_H
