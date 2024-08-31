#include "nd.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <iostream>
#include "xtensor/xadapt.hpp"
#include "xtensor/xview.hpp"
#include "xtl/xvariant.hpp"

using namespace godot;

void ND::_bind_methods() {
	godot::ClassDB::bind_static_method("ND", D_METHOD("asarray", "array"), &ND::asarray);
	godot::ClassDB::bind_static_method("ND", D_METHOD("zeros", "shape"), &ND::zeros);
	godot::ClassDB::bind_static_method("ND", D_METHOD("ones", "shape"), &ND::ones);

	godot::ClassDB::bind_static_method("ND", D_METHOD("add", "a", "b"), &ND::add);
	godot::ClassDB::bind_static_method("ND", D_METHOD("subtract", "a", "b"), &ND::subtract);
	godot::ClassDB::bind_static_method("ND", D_METHOD("multiply", "a", "b"), &ND::multiply);
	godot::ClassDB::bind_static_method("ND", D_METHOD("divide", "a", "b"), &ND::divide);
}

ND::ND() {
}

ND::~ND() {
	// Add your cleanup here.
}

bool _asshape(Variant shape, std::shared_ptr<xt::xarray<uint64_t>> &target) {
	auto type = shape.get_type();

	if (Variant::can_convert(type, Variant::Type::INT)) {
		target = std::make_shared<xt::xarray<uint64_t>>(xt::xarray<uint64_t> { uint64_t(shape) });
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_INT32_ARRAY)) {
		auto shape_array = PackedInt32Array(shape);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		target = std::make_shared<xt::xarray<uint64_t>>(
			xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape)
		);
		return true;
	}

	ERR_FAIL_V_MSG(false, "Variant cannot be converted to a shape.");
}

bool _asarray(Variant array, std::shared_ptr<NDArrayVariant> &target) {
	auto type = array.get_type();

	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>((Object*)(array))) {
			target = ndarray->array;
			return true;
		}
	}

	if (Variant::can_convert(type, Variant::Type::INT)) {
		// TODO Int array
		target = std::make_shared<NDArrayVariant>(xt::xarray<double>(uint64_t(array)));
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::FLOAT)) {
		target = std::make_shared<NDArrayVariant>(xt::xarray<double>(float_t(array)));
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_FLOAT32_ARRAY)) {
		auto shape_array = PackedInt32Array(array);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		target = std::make_shared<NDArrayVariant>(
			xt::xarray<double>(xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape))
		);
		return true;
	}

	ERR_FAIL_V_MSG(false, "Variant cannot be converted to an array.");
}

Variant ND::asarray(Variant array) {
	auto type = array.get_type();

	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>((Object*)(array))) {
			return array;
		}
	}

	NDArray *result = memnew(NDArray());
	if (!_asarray(array, result->array)) {
		return nullptr;
	}

	return Variant(result);
}

Variant ND::zeros(Variant shape) {
	std::shared_ptr<xt::xarray<uint64_t>> shape_array;
	if (!_asshape(shape, shape_array)) {
		return nullptr;
	}

	// General note: By creating the object first, and assigning later,
	//  we avoid creating the result on the stack first and copying to the heap later.
	// This means this kind of ugly contraption is quite a lot faster than the alternative.
	auto result = std::make_shared<NDArrayVariant>();
	xtl::get<xt::xarray<double>>(*result) = xt::zeros<double>(*shape_array);

	return Variant(memnew(NDArray(result)));
}

Variant ND::ones(Variant shape) {
	std::shared_ptr<xt::xarray<uint64_t>> shape_array;
	if (!_asshape(shape, shape_array)) {
		return nullptr;
	}

	auto result = std::make_shared<NDArrayVariant>();
	xtl::get<xt::xarray<double>>(*result) = xt::ones<double>(*shape_array);

	return Variant(memnew(NDArray(result)));
}

template <typename operation>
inline Variant bin_op(Variant a, Variant b) {
	std::shared_ptr<NDArrayVariant> a_;
	if (!_asarray(a, a_)) {
		return nullptr;
	}
	std::shared_ptr<NDArrayVariant> b_;
	if (!_asarray(b, b_)) {
		return nullptr;
	}

	auto result = std::make_shared<NDArrayVariant>();
	xtl::get<xt::xarray<double>>(*result) = operation()(xtl::get<xt::xarray<double>>(*a_), xtl::get<xt::xarray<double>>(*b_));

	return Variant(memnew(NDArray(result)));
}

Variant ND::add(Variant a, Variant b) {
	// godot::UtilityFunctions::print(xt::has_simd_interface<xt::xarray<int64_t>>::value);
	// godot::UtilityFunctions::print(xt::has_simd_type<xt::xarray<int64_t>>::value);
	return bin_op<std::plus<xt::xarray<double>>>(a, b);
}

Variant ND::subtract(Variant a, Variant b) {
	return bin_op<std::minus<xt::xarray<double>>>(a, b);
}

Variant ND::multiply(Variant a, Variant b) {
	return bin_op<std::multiplies<xt::xarray<double>>>(a, b);
}

Variant ND::divide(Variant a, Variant b) {
	return bin_op<std::divides<xt::xarray<double>>>(a, b);
}
