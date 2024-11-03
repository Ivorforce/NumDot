#ifndef STRIDE_TRICKS_HPP
#define STRIDE_TRICKS_HPP

#include <memory>
#include "varray.hpp"

namespace va {
	std::shared_ptr<VArray> as_strided(const VArray& array, const shape_type& shape, const strides_type& strides);
	std::shared_ptr<VArray> sliding_window_view(const VArray& array, const shape_type& window_shape);
	void convolve(VStoreAllocator& allocator, VArrayTarget target, const VArray& array, const VArray& kernel);
}

#endif //STRIDE_TRICKS_HPP
