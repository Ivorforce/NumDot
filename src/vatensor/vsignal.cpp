#include "vsignal.hpp"

#include "vcompute.hpp"
#include "xtensor-signal/fft.hpp"

void va::fft(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, std::ptrdiff_t axis) {
	va::xoperation_inplace<
		Feature::fft,
		promote::num_matching_complex_or_default_in_same_out<double_t>
	>(
		[axis](auto&& a) { return xt::fft::fft(std::forward<decltype(a)>(a), axis); },
		allocator,
		target,
		a.data
	);
}

void va::pad(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const std::vector<std::vector<std::size_t>>& pad_width, xt::pad_mode pad_mode, VScalar pad_value) {
	va::xoperation_inplace<
		Feature::pad,
		promote::common_in_same_out
	>(
		[&pad_width, pad_mode, pad_value](auto&& a) {
			using V = typename std::decay_t<decltype(a)>::value_type;
			return xt::pad(std::forward<decltype(a)>(a), pad_width, pad_mode, va::static_cast_scalar<V>(pad_value));
		},
		allocator,
		target,
		a.data
	);
}

void va::pad(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const std::vector<std::size_t>& pad_width, xt::pad_mode pad_mode, VScalar pad_value) {
	va::xoperation_inplace<
		Feature::pad,
		promote::common_in_same_out
	>(
		[&pad_width, pad_mode, pad_value](auto&& a) {
			using V = typename std::decay_t<decltype(a)>::value_type;
			return xt::pad(std::forward<decltype(a)>(a), pad_width, pad_mode, va::static_cast_scalar<V>(pad_value));
		},
		allocator,
		target,
		a.data
	);
}

void va::pad(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const size_t& pad_width, xt::pad_mode pad_mode, VScalar pad_value) {
	va::xoperation_inplace<
		Feature::pad,
		promote::common_in_same_out
	>(
		[&pad_width, pad_mode, pad_value](auto&& a) {
			using V = typename std::decay_t<decltype(a)>::value_type;
			return xt::pad(std::forward<decltype(a)>(a), pad_width, pad_mode, va::static_cast_scalar<V>(pad_value));
		},
		allocator,
		target,
		a.data
	);
}
