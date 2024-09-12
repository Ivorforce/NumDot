#ifndef NUMDOT_AS_ARRAY_H
#define NUMDOT_AS_ARRAY_H

#include <godot_cpp/variant/variant.hpp>  // for Variant
#include <variant>                        // for visit
#include "godot_cpp/variant/array.hpp"    // for Array
#include "vatensor/varray.h"                       // for to_compute_variant, VArray
#include "xtensor/xlayout.hpp"            // for layout_type

using namespace godot;

va::VArray variant_as_array(const Variant array);

template <typename P>
P xtvariant_to_packed(const va::VArray& array) {
	P p_array = P();

	std::visit([&p_array](auto carray){
		p_array.resize(carray.size());
		std::copy(carray.begin(), carray.end(), p_array.ptrw());
	}, array.to_compute_variant());

	return p_array;
}

Array xtvariant_to_godot_array(const va::VArray& array);

#endif
