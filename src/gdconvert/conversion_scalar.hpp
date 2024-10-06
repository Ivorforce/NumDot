#ifndef CONVERSION_SCALAR_HPP
#define CONVERSION_SCALAR_HPP

#include "vatensor/varray.hpp"            // for VScalar
#include "godot_cpp/variant/variant.hpp"  // for Variant

using namespace godot;

va::VScalar variant_to_vscalar(const Variant& variant);

#endif //CONVERSION_SCALAR_HPP
