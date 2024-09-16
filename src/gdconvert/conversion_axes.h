#ifndef CONVERSION_AXES_H
#define CONVERSION_AXES_H

#include <godot_cpp/variant/variant.hpp>
#include "vatensor/varray.h"

using namespace godot;

va::Axes variant_to_axes(const Variant& variant);

#endif //CONVERSION_AXES_H
