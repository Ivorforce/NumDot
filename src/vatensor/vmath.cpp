#include "vmath.hpp"

#include <utility>                                          // for move
#include "varray.hpp"                                // for VArray
#include "vcompute.hpp"                               // for XFunction
#include "vpromote.hpp"                                       // for num_funct...
#include "xtensor/xlayout.hpp"                              // for layout_type
#include "xtensor/xmath.hpp"                                // for maximum
#include "xtensor/xoperation.hpp"                           // for divides
#include "xtl/xfunctional.hpp"                              // for select
#include "scalar_tricks.hpp"

using namespace va;

template <typename A, typename B, typename C>
void clip(VStoreAllocator& allocator, const VArrayTarget& target, const A& a, const B& lo, const C& hi) {
	va::xoperation_inplace<
		Feature::clip,
		promote::reject_complex<promote::common_in_same_out>
	>(
		XFunction<xt::math::clamp_fun> {},
		allocator,
		target,
		a,
		lo,
		hi
	);
}

void va::clip(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& lo, const VData& hi) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	// TODO Check binary size add and perhaps just use min and max.

	if (va::dimension(lo) == 0 && va::dimension(hi) == 0) {
		::clip(allocator, target, a, to_single_value(lo), to_single_value(hi));
		return;
	}
#endif

	::clip(allocator, target, a, lo, hi);
}

void va::conjugate(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	if (!std::visit([](auto x) {
		using VTValue = typename std::decay_t<decltype(x)>::value_type;
		return xtl::is_complex<VTValue>::value;
	}, array)) {
		// It's not even a complex value; let's just assign in-place.
		if (va::dtype(array) == va::Bool) {
			// Need to make sure bools get int-ified.
			va::assign_cast(allocator, target, array, va::Int64);
		}
		else {
			va::assign(allocator, target, array);
		}

		return;
	}

	// This is technically just b = a; b[imag] = -a[imag]
	// But it may be slower to do it in 2 steps.
	// TODO We should find out how much space this saves, and if alloc - assign - assign_sub isn't actually faster.
	xoperation_inplace<
		Feature::negative,
		promote::reject_non_complex<promote::common_in_same_out>
	>(
		va::XFunction<xt::math::conj_fun> {},
		allocator,
		target,
		array
	);
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
