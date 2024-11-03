#include "vsignal.hpp"

#include "vcompute.hpp"
#include "xtensor-signal/fft.hpp"

void va::fft(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, std::ptrdiff_t axis) {
	va::xoperation_inplace<promote::num_matching_complex_or_default_in_same_out<double_t>>(
		[axis](auto&& a) { return xt::fft::fft(std::forward<decltype(a)>(a), axis); },
		allocator,
		target,
		a.data
	);
}
