#ifndef NUMDOT_AS_ARRAY_H
#define NUMDOT_AS_ARRAY_H

#include "vatensor/auto_defines.h"
#include <godot_cpp/variant/variant.hpp>  // for Variant
#include <variant>                        // for visit
#include "godot_cpp/variant/array.hpp"    // for Array
#include "vatensor/varray.h"              // for VArray

using namespace godot;

va::VArray variant_as_array(const Variant& array);
va::VArray variant_as_array(const Variant& array, va::DType dtype, bool copy);

template <typename P, typename A>
P packed_from_sequence(A& a) {
	P p;
	p.resize(a.size());
	std::copy(a.begin(), a.end(), p.ptrw());
	return p;
}

template <typename P>
P varray_to_packed(const va::VArray& array) {
	return std::visit([](auto carray){
		return packed_from_sequence<P>(carray);
	}, array.to_compute_variant());
}

void find_shape_and_dtype(va::shape_type& shape, va::DType &dtype, const Array& input_array);
Array varray_to_godot_array(const va::VArray& array);

#endif
