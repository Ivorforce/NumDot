#ifndef VSIGNAL_HPP
#define VSIGNAL_HPP

#include "varray.hpp"
#include "xtensor/xpad.hpp"

namespace va {
	void fft(VStoreAllocator& allocator, const VArrayTarget& target, const VArray& a, std::ptrdiff_t axis);
	std::shared_ptr<VArray> fft_freq(VStoreAllocator& allocator, std::size_t n, double_t d);

	void pad(VStoreAllocator& allocator, const VArrayTarget& target, const VArray& a, const std::vector<std::vector<std::size_t>> &pad_width, xt::pad_mode pad_mode, VScalar pad_value);
	void pad(VStoreAllocator& allocator, const VArrayTarget& target, const VArray& a, const std::vector<std::size_t> &pad_width, xt::pad_mode pad_mode, VScalar pad_value);
	void pad(VStoreAllocator& allocator, const VArrayTarget& target, const VArray& a, const size_t &pad_width, xt::pad_mode pad_mode, VScalar pad_value);
}

#endif //VSIGNAL_HPP
