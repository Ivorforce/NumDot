#ifndef NUMDOT_AS_SHAPE_H
#define NUMDOT_AS_SHAPE_H

#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "ndarray.h"

using namespace godot;

template <typename Sh, typename T>
Sh packed_as_shape(const T& shape_array) {
	Sh sh;
	sh.assign(shape_array.ptr(), shape_array.ptr() + shape_array.size());
	return sh;
}

template <typename C, typename T>
T variant_as_shape(const Variant shape) {
	auto type = shape.get_type();

	switch (type) {
		case Variant::OBJECT:
			if (auto ndarray = Object::cast_to<NDArray>(shape)) {
				// TODO
				// target = ndarray->array;
				throw std::runtime_error("Unsupported type");
			}
			break;
		case Variant::INT:
			return { C(int64_t(shape)) };
		case Variant::PACKED_BYTE_ARRAY:
			return packed_as_shape<T>(PackedByteArray(shape));
		case Variant::PACKED_INT32_ARRAY:
			return packed_as_shape<T>(PackedInt32Array(shape));
		case Variant::PACKED_INT64_ARRAY:
			return packed_as_shape<T>(PackedInt64Array(shape));
		case Variant::VECTOR2I: {
			auto vector = Vector2i(shape);
			return { C(vector.x), C(vector.y) };
		}
		case Variant::VECTOR3I: {
			auto vector = Vector3i(shape);
			return { C(vector.x), C(vector.y), C(vector.z) };
		}
		case Variant::VECTOR4I: {
			auto vector = Vector4i(shape);
			return { C(vector.x), C(vector.y), C(vector.z), C(vector.w) };
		}
		default:
			break;
	}

	// TODO Godot will probably convert float to int. We should check.
	if (Variant::can_convert(type, Variant::Type::PACKED_INT32_ARRAY)) {
		return packed_as_shape<T>(PackedInt32Array(shape));
	}

	throw std::runtime_error("Unsupported type");
}
#endif
