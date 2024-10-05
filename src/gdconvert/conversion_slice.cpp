#include "conversion_slice.hpp"

#include <cstdint>                            // for int64_t
#include <ndarray.hpp>
#include <ndutil.hpp>
#include <stdexcept>                          // for runtime_error
#include <vatensor/allocate.hpp>
#include "conversion_array.hpp"
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "godot_cpp/variant/variant.hpp"      // for Variant
#include "godot_cpp/variant/vector4i.hpp"     // for Vector4i
#include "xtensor/xslice.hpp"                 // for xrange_adaptor, xall_tag

using xtuph = xt::placeholders::xtuph;

xt::xstrided_slice<std::ptrdiff_t> variant_to_slice_part(const Variant& variant) {
	const auto type = variant.get_type();

	switch (type) {
		case Variant::VECTOR4I: {
			const auto vector = static_cast<Vector4i>(variant);
			switch (vector.x) {
				case 0b000: return { xt::xrange_adaptor { xtuph {}, xtuph {}, xtuph {} } };
				case 0b001: return { xt::xrange_adaptor { xtuph {}, xtuph {}, static_cast<std::ptrdiff_t>(vector.w) } };
				case 0b010: return { xt::xrange_adaptor { xtuph {}, static_cast<std::ptrdiff_t>(vector.z), xtuph {} } };
				case 0b011: return { xt::xrange_adaptor { xtuph {}, static_cast<std::ptrdiff_t>(vector.z), static_cast<std::ptrdiff_t>(vector.w) } };
				case 0b100: return { xt::xrange_adaptor { static_cast<std::ptrdiff_t>(vector.y), xtuph {}, xtuph {} } };
				case 0b101: return { xt::xrange_adaptor { static_cast<std::ptrdiff_t>(vector.y), xtuph {}, static_cast<std::ptrdiff_t>(vector.w) } };
				case 0b110: return { xt::xrange_adaptor { static_cast<std::ptrdiff_t>(vector.y), static_cast<std::ptrdiff_t>(vector.z), xtuph {} } };
				case 0b111: return { xt::xrange_adaptor { static_cast<std::ptrdiff_t>(vector.y), static_cast<std::ptrdiff_t>(vector.z), static_cast<std::ptrdiff_t>(vector.w) } };
				default: break;
			}
			break;
		}
		case Variant::NIL:
			return xt::all();
		case Variant::INT:
			return static_cast<int64_t>(variant);
		case Variant::STRING_NAME: {
			const auto string_name = static_cast<StringName>(variant);

			if (string_name == ::newaxis()) {
				return xt::newaxis();
			}
			else if (string_name == ::ellipsis()) {
				return xt::ellipsis();
			}
		}
		break;
		default:
			break;
	}

	throw std::runtime_error("Variant cannot be converted to a slice.");
}

SliceVariant variants_to_slice_variant(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	if (arg_count == 0)
		return nullptr;

	if (arg_count == 1) {
		const Variant& first_arg = *args[0];
		switch (first_arg.get_type()) {
			case Variant::OBJECT: {
				// May be mask access!
				if (const auto ndarray = Object::cast_to<NDArray>(first_arg)) {
					// Mask access?
					if (ndarray->array->dtype() == va::Bool) return SliceMask { ndarray->array };

					// Could be float, but we'll notice later.
					return SliceIndexList { ndarray->array };
				}
			}
			case Variant::ARRAY: {
				const auto input_varray = array_as_varray(first_arg);

				if (input_varray->dtype() == va::Bool)
					return SliceMask { input_varray };
				else
					return SliceIndexList { input_varray };
			}
			default:
				break;
		}
	}

	xt::xstrided_slice_vector sv(arg_count);
	for (int i = 0; i < arg_count; i++) {
		sv[i] = variant_to_slice_part(*args[i]);
	}
	return sv;
}
