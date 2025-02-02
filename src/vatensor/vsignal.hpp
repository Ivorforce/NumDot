#ifndef VSIGNAL_HPP
#define VSIGNAL_HPP

#include "varray.hpp"
#include "xtensor/xpad.hpp"

namespace va {
	std::shared_ptr<VArray> fft_freq(VStoreAllocator& allocator, std::size_t n, double_t d);
}

#endif //VSIGNAL_HPP
