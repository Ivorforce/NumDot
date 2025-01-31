#include "vmath.hpp"

#include "varray.hpp"                                // for VArray
#include "vcompute.hpp"                               // for XFunction
#include "vpromote.hpp"                                       // for num_funct...
#include "vfunc/entrypoints.hpp"
#include "xtensor/xlayout.hpp"                              // for layout_type
#include "xtensor/xmath.hpp"                                // for maximum

using namespace va;

void va::clip(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& lo, const VData& hi) {
	// We can just assign to target directly in the first call, and then use it as a temporary in the second call.
	// No invariants are destroyed with this assumption, and it saves us having to re-define a ternary function
	// or create a temporary variable.
	va::minimum(allocator, target, a, hi);
	va::maximum(allocator, target, unwrap_target(target), lo);
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
