#include "conversion_scalar.hpp"

#include "ndarray.hpp"

va::VScalar variant_to_vscalar(const Variant& variant) {
	switch (variant.get_type()) {
		case Variant::BOOL:
			return static_cast<bool>(variant);
		case Variant::INT:
			return static_cast<int64_t>(variant);
		case Variant::FLOAT:
			return static_cast<double_t>(variant);
		case Variant::ARRAY: {
			const Array array = variant;
			if (array.size() == 1) {
				return variant_to_vscalar(array[0]);
			}
		}
		case Variant::OBJECT: {
			if (const auto ndarray = Object::cast_to<NDArray>(variant)) {
				return ndarray->array->to_single_value();
			}
		}
		default:
			break;
	}

	throw std::runtime_error("Unsupported variant type");
}
