#ifndef NUMDOT_AS_SLICE_H
#define NUMDOT_AS_SLICE_H

#include <godot_cpp/godot.hpp>
#include "xtensor/xtensor.hpp"
#include "xtensor/xstrided_view.hpp"

using namespace godot;

static xt::xstrided_slice<std::ptrdiff_t> variant_as_slice_part(const Variant& variant) {
	auto type = variant.get_type();

	switch (type) {
		case Variant::NIL:
			return xt::all();
		case Variant::INT:
			return int64_t(variant);
		default:
			break;
	}

	throw std::runtime_error("Variant cannot be converted to a shape.");
}

#endif
