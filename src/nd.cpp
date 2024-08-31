#include "nd.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <iostream>
#include "xtensor/xadapt.hpp"
#include "xtl/xvariant.hpp"

using namespace godot;

void ND::_bind_methods() {
	godot::ClassDB::bind_static_method("ND", D_METHOD("asarray", "array"), &ND::asarray);
	godot::ClassDB::bind_static_method("ND", D_METHOD("zeros", "shape"), &ND::zeros);
	godot::ClassDB::bind_static_method("ND", D_METHOD("ones", "shape"), &ND::ones);

	godot::ClassDB::bind_static_method("ND", D_METHOD("add", "a", "b"), &ND::add);
}

ND::ND() {
}

ND::~ND() {
	// Add your cleanup here.
}

std::optional<xt::xarray<uint64_t>> ND::_asshape(Variant shape) {
	auto type = shape.get_type();

	if (Variant::can_convert(type, Variant::Type::INT)) {
		auto size = int64_t(shape);
		return { size };
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_INT32_ARRAY)) {
		auto shape_array = PackedInt32Array(shape);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		xt::xarray<uint64_t> shape = xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape);
		
		return shape;
	}

	ERR_FAIL_V_MSG(std::nullopt, "Variant cannot be converted to a shape.");
}

std::optional<NDArrayVariant> ND::_asarray(Variant array) {
	auto type = array.get_type();

	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>((Object*)(array))) {
			return ndarray->array;
		}
	}

	if (Variant::can_convert(type, Variant::Type::INT)) {
		// TODO Int array
		auto value = uint64_t(array);
		return xt::xarray<double>(value);
	}
	if (Variant::can_convert(type, Variant::Type::FLOAT)) {
		auto value = float_t(array);
		return xt::xarray<double>(value);
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_FLOAT32_ARRAY)) {
		auto shape_array = PackedInt32Array(array);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		xt::xarray<double> shape = xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape);
		
		return shape;
	}

	ERR_FAIL_V_MSG(std::nullopt, "Variant cannot be converted to an array.");
}

Variant ND::asarray(Variant array) {
	auto type = array.get_type();

	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>((Object*)(array))) {
			return array;
		}
	}

	if (auto converted = ND::_asarray(array)) {
		return Variant(memnew(NDArray(*converted)));
	}

	return nullptr;
}

Variant ND::zeros(Variant shape) {
	if (auto shape_array = _asshape(shape)) {
		xt::xarray<double> array = xt::zeros<double>(*shape_array);
		return Variant(memnew(NDArray(array)));
	}

	return nullptr;
}

Variant ND::ones(Variant shape) {
	if (auto shape_array = _asshape(shape)) {
		xt::xarray<double> array = xt::ones<double>(*shape_array);
		return Variant(memnew(NDArray(array)));
	}

	return nullptr;
}

Variant ND::add(Variant a, Variant b) {
	auto a_ = ND::_asarray(a);
	auto b_ = ND::_asarray(b);

	if (!a || !b) {
		return nullptr;
	}

	xt::xarray<double> result = xtl::get<xt::xarray<double>>(*a_) + xtl::get<xt::xarray<double>>(*b_);
	return Variant(memnew(NDArray(result)));
}

