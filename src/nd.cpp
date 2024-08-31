#include "nd.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <iostream>
#include "xtensor/xadapt.hpp"

using namespace godot;

void ND::_bind_methods() {
	godot::ClassDB::bind_static_method("ND", D_METHOD("zeros", "shape"), &ND::zeros, Variant());
}

ND::ND() {
}

ND::~ND() {
	// Add your cleanup here.
}

std::optional<xt::xarray<uint64_t>> ND::asshape(Variant shape) {
	if (shape.can_convert(shape.get_type(), Variant::Type::INT)) {
		auto size = int64_t(shape);
		return { size };
	}
	if (shape.can_convert(shape.get_type(), Variant::Type::PACKED_INT32_ARRAY)) {
		auto shape_array = PackedInt32Array(shape);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		xt::xarray<uint32_t> shape = xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape);
		
		return shape;
	}

	ERR_FAIL_V_MSG(std::nullopt, "Shape is not an array type.");
}

Variant ND::zeros(Variant shape) {
	if (auto shape_array = asshape(shape)) {
		xt::xarray<double> array = xt::zeros<double>(*shape_array);
		return Variant(memnew(NDArray(array)));
	}

	return nullptr;
}
