#ifndef NUMDOT_AS_SLICE_H
#define NUMDOT_AS_SLICE_H

#include <cstddef>                            // for ptrdiff_t
#include <godot_cpp/variant/variant.hpp>      // for Variant
#include "gdextension_interface.h"            // for GDExtensionCallError
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "xtensor/xstrided_view.hpp"          // for xstrided_slice, xstride...

using namespace godot;

StringName newaxis();
StringName ellipsis();

xt::xstrided_slice<std::ptrdiff_t> variant_as_slice_part(const Variant& variant);
xt::xstrided_slice_vector variants_as_slice_vector(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);

#endif
