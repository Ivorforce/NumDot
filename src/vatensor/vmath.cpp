#include "vmath.hpp"

#include "varray.hpp"                                // for VArray
#include "vcompute.hpp"                               // for XFunction
#include "vpromote.hpp"                                       // for num_funct...
#include "vfunc/entrypoints.hpp"
#include "xtensor/xlayout.hpp"                              // for layout_type
#include "xtensor/xmath.hpp"                                // for maximum

using namespace va;

void va::clip(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& lo, const VData& hi) {
	// TODO Re-evaluate if it's worth it to make it a ternary vfunc.
	// TODO It should also be possible to do this without a temp variable.
	std::shared_ptr<va::VArray> tmp;
	va::minimum(allocator, &tmp, a, hi);
	va::maximum(allocator, target, tmp->data, lo);
}

void va::a0xb1_minus_a1xb0(va::VStoreAllocator& allocator, const va::VArrayTarget& target, const va::VData& a, const va::VData& b, const std::ptrdiff_t i0, const std::ptrdiff_t i1) {
	va::xoperation_inplace<
		va::Feature::cross,
		va::promote::num_in_same_out
	>(
		[i0, i1](auto& a, auto& b) {
			return (xt::strided_view(a, { xt::ellipsis(), i0 }) * xt::strided_view(b, { xt::ellipsis(), i1 }))
			- (xt::strided_view(a, { xt::ellipsis(), i1 }) * xt::strided_view(b, { xt::ellipsis(), i0 }));
		},
		allocator,
		target,
		a,
		b
	);
}
