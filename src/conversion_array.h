#ifndef NUMDOT_AS_ARRAY_H
#define NUMDOT_AS_ARRAY_H

#include <godot_cpp/godot.hpp>

#include "xtensor/xtensor.hpp"
#include "xtensor/xadapt.hpp"

#include "ndarray.h"
#include "xtv.h"

using namespace godot;

template <typename C, typename T>
std::shared_ptr<xtv::XTVariant> packed_as_xarray(const T shape_array) {
	uint64_t size = shape_array.size();

	xt::static_shape<std::size_t, 1> shape_of_shape = { size };

	return std::make_shared<xtv::XTVariant>(
		xt::xarray<C>(xt::adapt(shape_array.ptr(), size, xt::no_ownership(), shape_of_shape))
	);
}

static std::shared_ptr<xtv::XTVariant> variant_as_array(const Variant array) {
	auto type = array.get_type();

	// TODO A bunch of interesting types are still missing
	switch (type) {
		case Variant::OBJECT:
			if (auto ndarray = Object::cast_to<NDArray>(array)) {
				return ndarray->array;
			}
			break;
		case Variant::INT:
			return std::make_shared<xtv::XTVariant>(xt::xarray<int64_t>(int64_t(array)));
		case Variant::FLOAT:
			return std::make_shared<xtv::XTVariant>(xt::xarray<double_t>(double_t(array)));
		case Variant::PACKED_BYTE_ARRAY:
			return packed_as_xarray<uint8_t>(PackedByteArray(array));
		case Variant::PACKED_INT32_ARRAY:
			return packed_as_xarray<int32_t>(PackedInt32Array(array));
		case Variant::PACKED_INT64_ARRAY:
			return packed_as_xarray<int64_t>(PackedInt64Array(array));
		case Variant::PACKED_FLOAT32_ARRAY:
			return packed_as_xarray<float_t>(PackedFloat32Array(array));
		case Variant::PACKED_FLOAT64_ARRAY:
			return packed_as_xarray<double_t>(PackedFloat64Array(array));
		case Variant::VECTOR2I: {
			auto vector = Vector2i(array);
			return std::make_shared<xtv::XTVariant>(xt::xarray<int64_t>(
				{ int64_t(vector.x), int64_t(vector.y) }
			));
		}
		case Variant::VECTOR3I: {
			auto vector = Vector3i(array);
			return std::make_shared<xtv::XTVariant>(xt::xarray<int64_t>(
				{ int64_t(vector.x), int64_t(vector.y), int64_t(vector.z) }
			));
		}
		case Variant::VECTOR4I: {
			auto vector = Vector4i(array);
			return std::make_shared<xtv::XTVariant>(xt::xarray<int64_t>(
				{ int64_t(vector.x), int64_t(vector.y), int64_t(vector.z), int64_t(vector.w) }
			));
		}
		case Variant::VECTOR2: {
			auto vector = Vector2(array);
			return std::make_shared<xtv::XTVariant>(xt::xarray<double_t>(
				{ double_t(vector.x), double_t(vector.y) }
			));
		}
		case Variant::VECTOR3: {
			auto vector = Vector3(array);
			return std::make_shared<xtv::XTVariant>(xt::xarray<double_t>(
				{ double_t(vector.x), double_t(vector.y), double_t(vector.z) }
			));
		}
		case Variant::VECTOR4: {
			auto vector = Vector4(array);
			return std::make_shared<xtv::XTVariant>(xt::xarray<double_t>(
				{ double_t(vector.x), double_t(vector.y), double_t(vector.z), double_t(vector.w) }
			));
		}

		default:
			break;
	}

	// Try float first. Int may be more lossy.
	if (Variant::can_convert(type, Variant::Type::FLOAT)) {
		return std::make_shared<xtv::XTVariant>(xt::xarray<double_t>(double_t(array)));
	}
	if (Variant::can_convert(type, Variant::Type::INT)) {
		return std::make_shared<xtv::XTVariant>(xt::xarray<int64_t>(int64_t(array)));
	}

	// TODO Godot will happily convert every number to float.
	// We should manually adapt and look through Array to find what its parts are.
	if (Variant::can_convert(type, Variant::Type::PACKED_FLOAT64_ARRAY)) {
		return packed_as_xarray<double_t>(PackedFloat64Array(array));
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_INT64_ARRAY)) {
		return packed_as_xarray<int64_t>(PackedInt64Array(array));
	}

	throw std::runtime_error("Unsupported type");
}

template <typename P>
P xtvariant_to_packed(xtv::XTVariant& array) {
	P p_array = P();

	std::visit([&p_array](auto array){
		p_array.resize(array.size());
		std::copy(array.begin(), array.end(), p_array.ptrw());
	}, array);

	return p_array;
}

Array xtvariant_to_godot_array(xtv::XTVariant& array) {
	Array godot_array = Array();

	std::visit([&godot_array](auto array){
		godot_array.resize(array.size());
		auto start = array.begin();

		for (size_t i = 0; i < array.size(); ++i) {
        	godot_array[i] = *(start + i);
		}
	}, array);

	return godot_array;
}

#endif
