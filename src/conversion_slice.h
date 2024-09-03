#ifndef NUMDOT_AS_SLICE_H
#define NUMDOT_AS_SLICE_H

#include <godot_cpp/godot.hpp>

#include "xtensor/xstrided_view.hpp"

#include "ndrange.h"
#include "nd.h"

using namespace godot;

// TODO It somehow doesn't work with a template function... :<
// Feel free to try yourself.
struct ToRangeVisitor {
	xt::xstrided_slice<std::ptrdiff_t> operator()(xt::placeholders::xtuph a, xt::placeholders::xtuph b, xt::placeholders::xtuph c) { return xt::range(a, b, c); };
	xt::xstrided_slice<std::ptrdiff_t> operator()(xt::placeholders::xtuph a, xt::placeholders::xtuph b, std::ptrdiff_t c) { return xt::range(a, b, c); };

	xt::xstrided_slice<std::ptrdiff_t> operator()(xt::placeholders::xtuph a, std::ptrdiff_t b, xt::placeholders::xtuph c) { return xt::range(a, b, c); };
	xt::xstrided_slice<std::ptrdiff_t> operator()(xt::placeholders::xtuph a, std::ptrdiff_t b, std::ptrdiff_t c) { return xt::range(a, b, c); };

	xt::xstrided_slice<std::ptrdiff_t> operator()(std::ptrdiff_t a, xt::placeholders::xtuph b, xt::placeholders::xtuph c) { return xt::range(a, b, c); };
	xt::xstrided_slice<std::ptrdiff_t> operator()(std::ptrdiff_t a, xt::placeholders::xtuph b, std::ptrdiff_t c) { return xt::range(a, b, c); };

	xt::xstrided_slice<std::ptrdiff_t> operator()(std::ptrdiff_t a, std::ptrdiff_t b, xt::placeholders::xtuph c) { return xt::range(a, b, c); };
	xt::xstrided_slice<std::ptrdiff_t> operator()(std::ptrdiff_t a, std::ptrdiff_t b, std::ptrdiff_t c) { return xt::range(a, b, c); };
};

static xt::xstrided_slice<std::ptrdiff_t> variant_as_slice_part(const Variant& variant) {
	auto type = variant.get_type();

	switch (type) {
		case Variant::OBJECT:
			if (auto ndrange = Object::cast_to<NDRange>(variant)) {
				return std::visit(ToRangeVisitor{}, ndrange->start, ndrange->stop, ndrange->step);
			}
			break;
		case Variant::NIL:
			return xt::all();
		case Variant::INT:
			return int64_t(variant);
		case Variant::STRING_NAME:
			if (StringName(variant) == newaxis()) {
				return xt::newaxis();
			}
			break;
		default:
			break;
	}

	throw std::runtime_error("Variant cannot be converted to a shape.");
}

#endif
