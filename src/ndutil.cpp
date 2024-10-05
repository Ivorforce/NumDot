#include "ndutil.hpp"

StringName newaxis() {
	const auto newaxis = StringName("newaxis");
	return newaxis;
}

StringName ellipsis() {
	const auto ellipsis = StringName("...");
	return ellipsis;
}

StringName no_value() {
	const auto no_value = StringName("no_value");
	return no_value;
}

bool is_no_value(const Variant& variant) {
	switch (variant.get_type()) {
		case Variant::STRING_NAME:
			if (static_cast<StringName>(variant) == ::no_value()) {
				return true;
			}
		default:
			return false;
	}
}
