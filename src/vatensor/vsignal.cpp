#include "vsignal.hpp"

#include "create.hpp"
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

std::shared_ptr<va::VArray> va::fft_freq(VStoreAllocator& allocator, std::size_t n, double_t d) {
	// From NumPy docs:
	// f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
	// f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

	auto array = va::empty(allocator, DType::Float64, shape_type { n });
	auto data = std::get<compute_case<double_t*>>(array->data);

	const auto center_idx = n / 2;
	const double_t factor = d * static_cast<double_t>(n);

	for (int i = 0; i < center_idx; ++i) data(i) = i / factor;
	for (int i = 1; i <= center_idx; ++i) data(n - i) = -i / factor;

	return array;
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
