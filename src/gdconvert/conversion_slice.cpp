#include "conversion_slice.hpp"

#include <cstdint>                            // for int64_t
#include <stdexcept>                          // for runtime_error
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "godot_cpp/variant/variant.hpp"      // for Variant
#include "godot_cpp/variant/vector4i.hpp"     // for Vector4i
#include "xtensor/xslice.hpp"                 // for xrange_adaptor, xall_tag

StringName newaxis() {
	const StringName newaxis = StringName("newaxis");
	return newaxis;
}

StringName ellipsis() {
	const StringName ellipsis = StringName("...");
	return ellipsis;
}

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
		case Variant::STRING_NAME:
			if (StringName(variant) == ::newaxis()) {
				return xt::newaxis();
			}
			else if (StringName(variant) == ::ellipsis()) {
				return xt::ellipsis();
			}
			break;
		default:
			break;
	}

	throw std::runtime_error("Variant cannot be converted to a slice.");
}

xt::xstrided_slice_vector variants_to_slice_vector(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	xt::xstrided_slice_vector sv(arg_count);
	for (int i = 0; i < arg_count; i++) {
		sv[i] = variant_to_slice_part(*args[i]);
	}
	return sv;
}
