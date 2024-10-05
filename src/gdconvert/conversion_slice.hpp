#ifndef NUMDOT_AS_SLICE_H
#define NUMDOT_AS_SLICE_H

#include <cstddef>                            // for ptrdiff_t
#include <variant>                            // for variant
#include <godot_cpp/variant/variant.hpp>      // for Variant
#include <vatensor/varray.hpp>
#include "gdextension_interface.h"            // for GDExtensionCallError
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "xtensor/xstrided_view.hpp"          // for xstrided_slice, xstride...

using namespace godot;

using SliceVariant = std::variant<
	std::nullptr_t,
	xt::xstrided_slice_vector,
	std::shared_ptr<va::VArray>
>;

xt::xstrided_slice<std::ptrdiff_t> variant_to_slice_part(const Variant& variant);
SliceVariant variants_to_slice_variant(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error);

#endif
