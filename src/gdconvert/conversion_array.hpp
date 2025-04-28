#ifndef NUMDOT_AS_ARRAY_H
#define NUMDOT_AS_ARRAY_H

#include <cstddef>                        // for size_t
#include <functional>                     // for multiplies
#include <godot_cpp/variant/variant.hpp>  // for Variant
#include <numeric>                        // for accumulate
#include <memory>                        // for shared_ptr
#include <utility>                        // for forward
#include <variant>                        // for visit
#include "godot_cpp/variant/array.hpp"    // for Array
#include "vatensor/varray.hpp"              // for shape_type, DType, VArray
#include "xtensor/containers/xadapt.hpp"             // for adapt
#include "xtensor/containers/xbuffer_adaptor.hpp"    // for no_ownership
#include "xtensor/core/xlayout.hpp"            // for layout_type

class NDArray;

using namespace godot;

std::shared_ptr<va::VArray> ndarray_as_dtype(const NDArray& ndarray, va::DType dtype);
std::shared_ptr<va::VArray> array_as_varray(const Array& array);
std::shared_ptr<va::VArray> variant_as_array(const Variant& array);
std::shared_ptr<va::VArray> variant_as_array(const Variant& array, va::DType dtype, bool copy);

std::vector<std::shared_ptr<va::VArray>> variant_to_vector(const Variant& array);

void find_shape_and_dtype_of_array(va::shape_type& shape, va::DType& dtype, const Array& input_array);
void find_shape_and_dtype(va::shape_type& shape, va::DType& dtype, const Variant& array);
Array varray_to_godot_array(const va::VArray& array);

#endif
