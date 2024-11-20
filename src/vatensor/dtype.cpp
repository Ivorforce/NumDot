#include "dtype.hpp"

va::VScalar va::static_cast_scalar(const VScalar v, const DType dtype) {
	return std::visit(
		[v](const auto t) -> va::VScalar {
			using T = std::decay_t<decltype(t)>;
			return va::static_cast_scalar<T>(v);
		}, dtype_to_variant(dtype)
	);
}
