#include "vmath.hpp"

#include "varray.hpp"                                // for VArray
#include "vcompute.hpp"                               // for XFunction
#include "vfunc/entrypoints.hpp"

using namespace va;

void va::clip(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& lo, const VData& hi) {
	// TODO Re-evaluate if it's worth it to make it a ternary vfunc.
	// TODO It should also be possible to do this without a temp variable.
	std::shared_ptr<va::VArray> tmp;
	va::minimum(allocator, &tmp, a, hi);
	va::maximum(allocator, target, tmp->data, lo);
}
