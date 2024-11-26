#ifndef TRIGONOMETRY_H
#define TRIGONOMETRY_H

#include "varray.hpp"
#include "rearrange.hpp"
#include "ufunc/ufunc_features.hpp"

namespace va {
	static void angle(VStoreAllocator& allocator, const VArrayTarget& target, const std::shared_ptr<VArray>& array) {
		va::atan2(allocator, target, va::imag(array)->data, va::real(array)->data);
	}
}

#endif //TRIGONOMETRY_H
