#ifndef NUMDOT_AS_ARRAY_H
#define NUMDOT_AS_ARRAY_H

#include <godot_cpp/godot.hpp>

#include "xtensor/xtensor.hpp"
#include "xtensor/xadapt.hpp"

#include "varray.h"
#include "vcompute.h"

#include "ndarray.h"

using namespace godot;

template <typename C, typename T>
va::VArray packed_as_xarray(const T shape_array) {
	uint64_t size = shape_array.size();

	xt::static_shape<std::size_t, 1> shape_of_shape = { size };

	auto store = std::make_shared<xt::xarray<C>>(
		xt::xarray<C>(xt::adapt(shape_array.ptr(), size, xt::no_ownership(), shape_of_shape))
	);

	return va::from_store(store);
}

static va::VArray variant_as_array(const Variant array) {
	auto type = array.get_type();

	// TODO A bunch of interesting types are still missing
	switch (type) {
		case Variant::OBJECT:
			if (auto ndarray = Object::cast_to<NDArray>(array)) {
				return ndarray->array;
			}
			break;
		case Variant::INT: {
			auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(int64_t(array)));
			return va::from_store(store);
		}
		case Variant::FLOAT: {
			auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(double_t(array)));
			return va::from_store(store);
		}
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
			auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(
				{ int64_t(vector.x), int64_t(vector.y) }
			));
			return va::from_store(store);
		}
		case Variant::VECTOR3I: {
			auto vector = Vector3i(array);
			auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(
				{ int64_t(vector.x), int64_t(vector.y), int64_t(vector.z) }
			));
			return va::from_store(store);
		}
		case Variant::VECTOR4I: {
			auto vector = Vector4i(array);
			auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(
				{ int64_t(vector.x), int64_t(vector.y), int64_t(vector.z), int64_t(vector.w) }
			));
			return va::from_store(store);
		}
		case Variant::VECTOR2: {
			auto vector = Vector2(array);
			auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(
				{ double_t(vector.x), double_t(vector.y) }
			));
			return va::from_store(store);
		}
		case Variant::VECTOR3: {
			auto vector = Vector3(array);
			auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(
				{ double_t(vector.x), double_t(vector.y), double_t(vector.z) }
			));
			return va::from_store(store);
		}
		case Variant::VECTOR4: {
			auto vector = Vector4(array);
			auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(
				{ double_t(vector.x), double_t(vector.y), double_t(vector.z), double_t(vector.w) }
			));
			return va::from_store(store);
		}

		default:
			break;
	}

	// Try float first. Int may be more lossy.
	if (Variant::can_convert(type, Variant::Type::FLOAT)) {
		auto store = std::make_shared<xt::xarray<double_t>>(xt::xarray<double_t>(double_t(array)));
		return va::from_store(store);
	}
	if (Variant::can_convert(type, Variant::Type::INT)) {
		auto store = std::make_shared<xt::xarray<int64_t>>(xt::xarray<int64_t>(int64_t(array)));
		return va::from_store(store);
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
P xtvariant_to_packed(const va::VArray& array) {
	P p_array = P();

	std::visit([&p_array](auto carray){
		p_array.resize(carray.size());
		std::copy(carray.begin(), carray.end(), p_array.ptrw());
	}, va::to_compute_variant(array));

	return p_array;
}

static Array xtvariant_to_godot_array(const va::VArray& array) {
	Array godot_array = Array();

	std::visit([&godot_array](auto carray){
		godot_array.resize(carray.size());
		auto start = carray.begin();

		for (size_t i = 0; i < carray.size(); ++i) {
        	godot_array[i] = *(start + i);
		}
	}, to_compute_variant(array));

	return godot_array;
}

#endif
