#ifndef NUMDOT_AS_SLICE_H
#define NUMDOT_AS_SLICE_H

#include <cstddef>                            // for ptrdiff_t
#include <variant>                            // for variant
#include <godot_cpp/variant/variant.hpp>      // for Variant
#include <vatensor/varray.hpp>
#include "gdextension_interface.h"            // for GDExtensionCallError
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "xtensor/views/xstrided_view.hpp"          // for xstrided_slice, xstride...

using namespace godot;

struct SliceMask { std::shared_ptr<va::VArray> mask; };
struct SliceIndexList { std::shared_ptr<va::VArray> index_list; };

using single_axis_slice = std::tuple<xt::xstrided_slice<std::ptrdiff_t>, std::ptrdiff_t>;

using SliceVariant = std::variant<
	std::nullptr_t,
	single_axis_slice,
	xt::xstrided_slice_vector,
	SliceMask,
	SliceIndexList
>;

xt::xstrided_slice<std::ptrdiff_t> variant_to_slice_part(const Variant& variant);
SliceVariant variants_to_slice_variant(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error);
std::optional<va::axes_type> slice_vector_to_axes_list(const xt::xstrided_slice_vector& vector);

#endif
