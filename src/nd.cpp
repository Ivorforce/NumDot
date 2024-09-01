#include "nd.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <iostream>
#include "xtensor/xadapt.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xlayout.hpp"

#include "xtv.h"

using namespace godot;

void ND::_bind_methods() {
	godot::ClassDB::bind_static_method("ND", D_METHOD("asarray", "array", "dtype"), &ND::asarray, DEFVAL(nullptr), DEFVAL(NDArray::DType::DTypeMax));
	godot::ClassDB::bind_static_method("ND", D_METHOD("array", "array", "dtype"), &ND::array, DEFVAL(nullptr), DEFVAL(NDArray::DType::DTypeMax));
	godot::ClassDB::bind_static_method("ND", D_METHOD("zeros", "shape", "dtype"), &ND::zeros, DEFVAL(nullptr), DEFVAL(NDArray::DType::Double));
	godot::ClassDB::bind_static_method("ND", D_METHOD("ones", "shape", "dtype"), &ND::ones, DEFVAL(nullptr), DEFVAL(NDArray::DType::Double));

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

template <typename T>
bool _asshape(Variant shape, T &target) {
	auto type = shape.get_type();

	if (Variant::can_convert(type, Variant::Type::INT)) {
		target = { uint64_t(shape) };
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_INT32_ARRAY)) {
		auto shape_array = PackedInt32Array(shape);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		target = xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape);
		return true;
	}

	ERR_FAIL_V_MSG(false, "Variant cannot be converted to a shape.");
}

bool _asarray(Variant array, std::shared_ptr<xtv::Variant> &target) {
	auto type = array.get_type();

	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>((Object*)(array))) {
			target = ndarray->array;
			return true;
		}
	}

	if (Variant::can_convert(type, Variant::Type::INT)) {
		// TODO Int array
		target = std::make_shared<xtv::Variant>(xt::xarray<double>());
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::FLOAT)) {
		target = std::make_shared<xtv::Variant>(xt::xarray<double>(float_t(array)));
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_FLOAT32_ARRAY)) {
		auto shape_array = PackedInt32Array(array);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		target = std::make_shared<xtv::Variant>(
			xt::xarray<double>(xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape))
		);
		return true;
	}

	ERR_FAIL_V_MSG(false, "Variant cannot be converted to an array.");
}

Variant ND::asarray(Variant array, xtv::DType dtype) {
	auto type = array.get_type();

	// Can we take a view?
	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>((Object*)(array))) {
			if (dtype == xtv::DType::DTypeMax || ndarray->dtype() == dtype) {
				return array;
			}
		}
	}

	// Ok, we need a copy of the data.
	return ND::array(array, dtype);
}

Variant ND::array(Variant array, xtv::DType dtype) {
	auto type = array.get_type();

	std::shared_ptr<xtv::Variant> existing_array;
	if (!_asarray(array, existing_array)) {
		return nullptr;
	}

	if (dtype == xtv::DType::DTypeMax) {
		dtype = xtv::DType((*existing_array).index());
	}

	auto result = xtv::array(*existing_array, dtype);
	if (result == nullptr) {
		ERR_FAIL_V_MSG(nullptr, "Dtype must be set for this operation.");\
	}
	
	return Variant(memnew(NDArray(result)));
}

Variant ND::zeros(Variant shape, xtv::DType dtype) {
	xt::xarray<size_t> shape_array;
	if (!_asshape(shape, shape_array)) {
		return nullptr;
	}

	auto result = xtv::zeros(shape_array, dtype);
	if (result == nullptr) {
		ERR_FAIL_V_MSG(nullptr, "Dtype must be set for this operation.");\
	}
	
	return Variant(memnew(NDArray(result)));
}

Variant ND::ones(Variant shape, xtv::DType dtype) {
	xt::xarray<size_t> shape_array;
	if (!_asshape(shape, shape_array)) {
		return nullptr;
	}

	auto result = xtv::ones(shape_array, dtype);
	if (result == nullptr) {
		ERR_FAIL_V_MSG(nullptr, "Dtype must be set for this operation.");\
	}
	
	return Variant(memnew(NDArray(result)));
}

template <typename operation>
inline Variant binOp(Variant a, Variant b) {
	std::shared_ptr<xtv::Variant> a_;
	if (!_asarray(a, a_)) {
		return nullptr;
	}
	std::shared_ptr<xtv::Variant> b_;
	if (!_asarray(b, b_)) {
		return nullptr;
	}

	return Variant(memnew(NDArray(xtv::operation<operation>(*a_, *b_))));
}

Variant ND::add(Variant a, Variant b) {
	// godot::UtilityFunctions::print(xt::has_simd_interface<xt::xarray<int64_t>>::value);
	// godot::UtilityFunctions::print(xt::has_simd_type<xt::xarray<int64_t>>::value);
	return binOp<xtv::Add>(a, b);
}

Variant ND::subtract(Variant a, Variant b) {
	return binOp<xtv::Subtract>(a, b);
}

Variant ND::multiply(Variant a, Variant b) {
	return binOp<xtv::Multiply>(a, b);
}

Variant ND::divide(Variant a, Variant b) {
	return binOp<xtv::Divide>(a, b);
}
