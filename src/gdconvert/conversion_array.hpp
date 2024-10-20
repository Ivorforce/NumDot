#ifndef NUMDOT_AS_ARRAY_H
#define NUMDOT_AS_ARRAY_H

#include "vatensor/auto_defines.hpp"

#include <cstddef>                        // for size_t
#include <functional>                     // for multiplies
#include <godot_cpp/variant/variant.hpp>  // for Variant
#include <numeric>                        // for accumulate
#include <memory>                        // for shared_ptr
#include <utility>                        // for forward
#include <variant>                        // for visit
#include <vatensor/vpromote.hpp>

#include "godot_cpp/variant/array.hpp"    // for Array
#include "vatensor/varray.hpp"              // for shape_type, DType, VArray
#include "xtensor/xadapt.hpp"             // for adapt
#include "xtensor/xbuffer_adaptor.hpp"    // for no_ownership
#include "xtensor/xlayout.hpp"            // for layout_type

class NDArray;

using namespace godot;

std::shared_ptr<va::VArray> ndarray_as_dtype(const NDArray& ndarray, va::DType dtype);
std::shared_ptr<va::VArray> array_as_varray(const Array& array);
std::shared_ptr<va::VArray> variant_as_array(const Variant& array);
std::shared_ptr<va::VArray> variant_as_array(const Variant& array, va::DType dtype, bool copy);

std::vector<std::shared_ptr<va::VArray>> variant_to_vector(const Variant& array);

template<typename T, typename Compute>
void fill_c_array_flat(T target, const Compute& carray) {
	if constexpr (!std::is_convertible_v<va::promote::value_type_v<std::decay_t<Compute>>, std::decay_t<T>>) {
		throw std::runtime_error("Cannot promote in this way.");
	}
	else {
		std::copy(carray.begin(), carray.end(), target);
	}
}

template<typename T>
void fill_c_array_flat(T* target, const va::VRead& array) {
	std::visit(
		[target](auto& carray) {
            fill_c_array_flat(target, carray);
		}, array
	);
}

template<typename T>
auto adapt_c_array(T&& ptr, const va::shape_type& shape) {
	const auto size = std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<>());
	return xt::adapt<xt::layout_type::dynamic, T, xt::no_ownership, va::shape_type>(
		std::forward<T>(ptr), size, xt::no_ownership(), shape, xt::layout_type::row_major
	);
}

void find_shape_and_dtype_of_array(va::shape_type& shape, va::DType& dtype, const Array& input_array);
void find_shape_and_dtype(va::shape_type& shape, va::DType& dtype, const Variant& array);
Array varray_to_godot_array(const va::VArray& array);

#endif
