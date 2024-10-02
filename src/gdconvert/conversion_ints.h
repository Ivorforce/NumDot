#ifndef NUMDOT_AS_SHAPE_H
#define NUMDOT_AS_SHAPE_H

#include <godot_cpp/variant/variant.hpp>  // for Variant
#include "gdextension_interface.h"        // for GDExtensionCallError, GDExt...
#include "vatensor/varray.h"              // for axes_type, shape_type

using namespace godot;

template<typename T, typename Packed>
T packed_as_array(Packed packed) {
	T axes;
	axes.assign(packed.ptr(), packed.ptr() + packed.size());
	return axes;
}

va::shape_type variant_to_shape(const Variant& variant);
va::axes_type variant_to_axes(const Variant& variant);
va::axes_type variants_to_axes(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error);

#endif
