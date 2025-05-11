#include "ndutil.hpp"

StringName axis_all() {
	static const auto axis_all = StringName(":", true);
	return axis_all;
}

StringName newaxis() {
	static const auto newaxis = StringName("newaxis", true);
	return newaxis;
}

StringName ellipsis() {
	static const auto ellipsis = StringName("...", true);
	return ellipsis;
}

StringName no_value() {
	static const auto no_value = StringName("no_value", true);
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
