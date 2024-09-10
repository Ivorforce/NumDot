#ifndef NUMDOT_AS_SLICE_H
#define NUMDOT_AS_SLICE_H

#include <godot_cpp/godot.hpp>

#include "xtensor/xstrided_view.hpp"

#include "ndrange.h"
#include "nd.h"

using namespace godot;

// TODO Somehow, I can't manage to make this a lambda function visitor.
struct ToRangeVisitor {
    template <typename T1, typename T2, typename T3>
    xt::xstrided_slice<std::ptrdiff_t> operator()(T1 a, T2 b, T3 c) {
        return xt::range(a, b, c);
    }
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
			if (StringName(variant) == ::newaxis()) {
				return xt::newaxis();
			}
			break;
		default:
			break;
	}

	throw std::runtime_error("Variant cannot be converted to a shape.");
}

static xt::xstrided_slice_vector variants_as_slice_vector(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	xt::xstrided_slice_vector sv(arg_count);
	for (int i = 0; i < arg_count; i++) {
		sv[i] = variant_as_slice_part(*args[i]);
	}
	return sv;
}

#endif
