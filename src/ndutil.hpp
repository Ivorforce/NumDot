#ifndef NDUTIL_HPP
#define NDUTIL_HPP

#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "godot_cpp/variant/variant.hpp"  // for Variant

using namespace godot;

StringName newaxis();
StringName ellipsis();
StringName no_value();

bool is_no_value(const Variant& variant);

#endif //NDUTIL_HPP
