#ifndef ALLOCATE_H
#define ALLOCATE_H

#include "auto_defines.hpp"
#include "varray.hpp"

namespace va {
	std::shared_ptr<VArray> full(VScalar fill_value, const shape_type& shape);
	std::shared_ptr<VArray> empty(DType dtype, const shape_type& shape);
	std::shared_ptr<VArray> eye(DType dtype, const shape_type& shape, int k);

	std::shared_ptr<VArray> copy(const VRead& read);
	std::shared_ptr<VArray> copy_as_dtype(const VRead& other, DType dtype);

	std::shared_ptr<VArray> tile(const VArray& array, const shape_type& reps, bool inner);
}

#endif //ALLOCATE_H
