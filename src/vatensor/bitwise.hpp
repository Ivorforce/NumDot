#ifndef VATENSOR_BITWISE_HPP
#define VATENSOR_BITWISE_HPP

#include "varray.hpp"

namespace va {
	void bitwise_and(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b);
	void bitwise_or(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b);
	void bitwise_xor(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b);
	void bitwise_not(VStoreAllocator& allocator, VArrayTarget target, const VData& a);
	void bitwise_left_shift(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b);
	void bitwise_right_shift(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b);
}
#endif //VATENSOR_BITWISE_HPP
