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

std::vector<size_t> variant_as_shape(Variant shape);
va::strides_type variant_as_strides(Variant shape);

#endif
