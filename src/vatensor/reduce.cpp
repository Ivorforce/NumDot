#include "reduce.hpp"

#include "rearrange.hpp"
#include "create.hpp"
#include "varray.hpp"                            // for VArray, axes_...
#include "vcompute.hpp"                                   // for vreduce, xope...
#include "vpromote.hpp"                                   // for num_matching_...
#include "vfunc/entrypoints.hpp"

void va::count_nonzero(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array, const axes_type* axes) {
	if (va::dtype(array) == va::Bool)
		return va::sum(allocator, target, array, axes);

	const auto is_nonzero = va::copy_as_dtype(allocator, array, va::Bool);
	return va::sum(allocator, target, is_nonzero->data, axes);
}

void va::trace(VStoreAllocator& allocator, const VArrayTarget& target, const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2) {
	const auto diagonal = va::diagonal(varray, offset, axis1, axis2);
	const axes_type strides {-1};
	va::sum(allocator, target, diagonal->data, &strides);
}
