#ifndef NUMDOT_AS_SHAPE_H
#define NUMDOT_AS_SHAPE_H

#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "ndarray.h"

using namespace godot;

template <typename T, typename Sh>
inline bool packed_as_shape(const T& shape_array, Sh &target) {
	target.assign(shape_array.ptr(), shape_array.ptr() + shape_array.size());
	return true;
}

template <typename C, typename T>
bool variant_as_shape(const Variant shape, T &target) {
	auto type = shape.get_type();

	switch (type) {
		case Variant::OBJECT:
			if (auto ndarray = Object::cast_to<NDArray>(shape)) {
				// TODO
				// target = ndarray->array;
				return false;
			}
			break;
		case Variant::INT:
			target = { C(int64_t(shape)) };
			return true;
		case Variant::PACKED_BYTE_ARRAY:
			return packed_as_shape(PackedByteArray(shape), target);
		case Variant::PACKED_INT32_ARRAY:
			return packed_as_shape(PackedInt32Array(shape), target);
		case Variant::PACKED_INT64_ARRAY:
			return packed_as_shape(PackedInt64Array(shape), target);
		case Variant::VECTOR2I: {
			auto vector = Vector2i(shape);
			target = { C(vector.x), C(vector.y) };
			return true;
		}
		case Variant::VECTOR3I: {
			auto vector = Vector3i(shape);
			target = { C(vector.x), C(vector.y), C(vector.z) };
			return true;
		}
		case Variant::VECTOR4I: {
			auto vector = Vector4i(shape);
			target = { C(vector.x), C(vector.y), C(vector.z), C(vector.w) };
			return true;
		}
		default:
			break;
	}

	// TODO Godot will probably convert float to int. We should check.
	if (Variant::can_convert(type, Variant::Type::PACKED_INT32_ARRAY)) {
		return packed_as_shape(PackedInt32Array(shape), target);
	}

	ERR_FAIL_V_MSG(false, "Variant cannot be converted to a shape.");
}
#endif
