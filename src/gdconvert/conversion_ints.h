#ifndef NUMDOT_AS_SHAPE_H
#define NUMDOT_AS_SHAPE_H

#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "ndarray.h"

using namespace godot;

template <typename T, typename Packed>
T packed_as_array(Packed packed) {
    T axes;
    axes.assign(packed.ptr(), packed.ptr() + packed.size());
    return axes;
}

std::vector<size_t> variant_to_shape(const Variant &shape);
va::strides_type variant_to_strides(const Variant &shape);
va::Axes variant_to_axes(const Variant &shape);

#endif
