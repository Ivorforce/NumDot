#include "conversion_array.hpp"

#include <vatensor/create.hpp>                         // for copy_as_dtype
#include <vatensor/vassign.hpp>                          // for assign
#include <cmath>                                       // for double_t, float_t
#include <cstddef>                                     // for size_t
#include <cstdint>                                     // for int64_t, int32_t
#include <iterator>                                    // for size
#include <stdexcept>                                   // for runtime_error
#include <tuple>                                       // for tuple, make_tuple
#include <utility>                                     // for move
#include <vector>                                      // for allocator, vector
#include <vatensor/vcarray.hpp>
#include <vatensor/xtensor_store.hpp>
#include <vatensor/xscalar_store.hpp>
#include <vatensor/dtype.hpp>
#include <vatensor/vfunc/entrypoints.hpp>

#include "godot_cpp/classes/object.hpp"                // for Object
#include "godot_cpp/core/defs.hpp"                     // for real_t
#include "godot_cpp/core/object.hpp"                   // for Object::cast_to
#include "godot_cpp/variant/color.hpp"                 // for Color
#include "godot_cpp/variant/packed_byte_array.hpp"     // for PackedByteArray
#include "godot_cpp/variant/packed_color_array.hpp"    // for PackedColorArray
#include "godot_cpp/variant/packed_float32_array.hpp"  // for PackedFloat32A...
#include "godot_cpp/variant/packed_float64_array.hpp"  // for PackedFloat64A...
#include "godot_cpp/variant/packed_int32_array.hpp"    // for PackedInt32Array
#include "godot_cpp/variant/packed_int64_array.hpp"    // for PackedInt64Array
#include "godot_cpp/variant/packed_vector2_array.hpp"  // for PackedVector2A...
#include "godot_cpp/variant/packed_vector3_array.hpp"  // for PackedVector3A...
#include "godot_cpp/variant/packed_vector4_array.hpp"  // for PackedVector4A...
#include "godot_cpp/variant/variant.hpp"               // for Variant
#include "godot_cpp/variant/vector2.hpp"               // for Vector2
#include "godot_cpp/variant/vector2i.hpp"              // for Vector2i
#include "godot_cpp/variant/vector3.hpp"               // for Vector3
#include "godot_cpp/variant/vector3i.hpp"              // for Vector3i
#include "godot_cpp/variant/vector4.hpp"               // for Vector4
#include "godot_cpp/variant/vector4i.hpp"              // for Vector4i
#include "ndarray.hpp"                                   // for NDArray
#include "tensor_fixed_store.hpp"
#include "packed_array_store.hpp"
#include "variant_tensor.hpp"
#include "xtensor/containers/xarray.hpp"                          // for xarray_adaptor
#include "xtensor/containers/xbuffer_adaptor.hpp"                 // for xbuffer_adaptor
#include "xtensor/core/xlayout.hpp"                         // for layout_type
#include "xtensor/containers/xstorage.hpp"                        // for svector, uvector
#include "xtensor/views/xstrided_view.hpp"                   // for xstrided_slice...

void add_size_at_idx(va::shape_type& shape, const std::size_t idx, const std::size_t value) {
	if (shape.size() > idx) {
		const auto current_dim_size = shape[idx];

		// Sizes are the same.
		if (current_dim_size == value) return;

		// We defined the size before, this element can broadcast.
		if (value == 1 && current_dim_size >= 1) return;

		// We broadcasted before, this element defines the size.
		if (current_dim_size == 1 && value > 1) {
			shape[idx] = value;
			return;
		}

		throw std::runtime_error("array has an inhomogenous shape");
	}
	else if (shape.size() == idx) {
		shape.push_back(value);
	}
	else {
		throw std::runtime_error("index out of range");
	}
}

void find_shape_and_dtype_of_array(va::shape_type& shape, va::DType& dtype, const Array& input_array) {
	if (!va::is_any_dtype(dtype)) throw std::runtime_error("Invalid dtype.");

	std::vector<Array> current_dim_arrays = { input_array };

	for (std::size_t current_dim_idx = 0; true; ++current_dim_idx) {
		for (const auto& array : current_dim_arrays) {
			add_size_at_idx(shape, current_dim_idx, array.size());
		}

		std::vector<Array> next_dim_arrays;
		for (const Array& array : current_dim_arrays) {
			for (int i = 0; i < array.size(); ++i) {
				const Variant& array_element = array[i];

				switch (array_element.get_type()) {
					case Variant::OBJECT: {
						if (const auto ndarray = Object::cast_to<NDArray>(array_element)) {
							dtype = va::dtype_common_type_unchecked(dtype, ndarray->array->dtype());
							auto varray_dim_idx = current_dim_idx + 1;
							for (const auto size : ndarray->array->shape()) {
								add_size_at_idx(shape, varray_dim_idx, size);
								varray_dim_idx++;
							}
							continue;
						}
						break;
					}
					case Variant::ARRAY:
						next_dim_arrays.push_back(array_element);
						continue;
					case Variant::BOOL:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<bool>());
						continue;
					case Variant::INT:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<int64_t>());
						continue;
					case Variant::FLOAT:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<double_t>());
						continue;
					case Variant::PACKED_BYTE_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<uint8_t>());
						PackedByteArray packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_INT32_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<int32_t>());
						PackedInt32Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_INT64_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<int64_t>());
						PackedInt64Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_FLOAT32_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<float_t>());
						PackedFloat32Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_FLOAT64_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<double_t>());
						PackedFloat64Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_VECTOR2_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<real_t>());
						PackedVector2Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						add_size_at_idx(shape, current_dim_idx + 2, 2);
						continue;
					}
					case Variant::PACKED_VECTOR3_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<real_t>());
						PackedVector3Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						add_size_at_idx(shape, current_dim_idx + 2, 3);
						continue;
					}
					case Variant::PACKED_VECTOR4_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<real_t>());
						PackedVector4Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						add_size_at_idx(shape, current_dim_idx + 2, 4);
						continue;
					}
					case Variant::PACKED_COLOR_ARRAY: {
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<float_t>());
						PackedColorArray packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						add_size_at_idx(shape, current_dim_idx + 2, 4);
						continue;
					}
					case Variant::VECTOR2I:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<int64_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 2);
						continue;
					case Variant::VECTOR3I:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<int64_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 3);
						continue;
					case Variant::VECTOR4I:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<int64_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 4);
						continue;
					case Variant::VECTOR2:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<real_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 2);
						continue;
					case Variant::VECTOR3:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<real_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 3);
						continue;
					case Variant::COLOR:
						dtype = va::dtype_common_type(dtype, va::dtype_of_type<float_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 4);
						continue;
					case Variant::VECTOR4:
					case Variant::QUATERNION:
					case Variant::PLANE:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<real_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 4);
						continue;
					case Variant::BASIS:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<real_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 3);
						add_size_at_idx(shape, current_dim_idx + 2, 3);
						continue;
					case Variant::PROJECTION:
						dtype = va::dtype_common_type_unchecked(dtype, va::dtype_of_type<real_t>());
						add_size_at_idx(shape, current_dim_idx + 1, 4);
						add_size_at_idx(shape, current_dim_idx + 2, 4);
						continue;
					default:
						break;
				}

				throw std::runtime_error("unsupported array type");
			}
		}

		if (next_dim_arrays.empty()) break;

		current_dim_arrays = std::move(next_dim_arrays);
	}
}

void find_shape_and_dtype(va::shape_type& shape, va::DType& dtype, const Variant& array) {
	switch (array.get_type()) {
		case Variant::OBJECT: {
			if (const auto ndarray = Object::cast_to<NDArray>(array)) {
				shape = ndarray->array->shape();
				dtype = ndarray->array->dtype();
				return;
			}
			break;
		}
		case Variant::ARRAY: {
			find_shape_and_dtype_of_array(shape, dtype, array);
			return;
		}
		case Variant::BOOL: {
			shape = { 1 };
			dtype = va::DType::Bool;
			return;
		}
		case Variant::INT: {
			shape = { 1 };
			dtype = va::DType::Int64;
			return;
		}
		case Variant::FLOAT: {
			shape = { 1 };
			dtype = va::DType::Float64;
			return;
		}
		case Variant::PACKED_BYTE_ARRAY: {
			const auto packed = PackedByteArray(array);
			shape = { static_cast<std::size_t>(packed.size()) };
			dtype = va::DType::UInt8;
			return;
		}
		case Variant::PACKED_INT32_ARRAY: {
			const auto packed = PackedInt32Array(array);
			shape = { static_cast<std::size_t>(packed.size()) };
			dtype = va::DType::Int32;
			return;
		}
		case Variant::PACKED_INT64_ARRAY: {
			const auto packed = PackedInt64Array(array);
			shape = { static_cast<std::size_t>(packed.size()) };
			dtype = va::DType::Int64;
			return;
		}
		case Variant::PACKED_FLOAT32_ARRAY: {
			const auto packed = PackedFloat32Array(array);
			shape = { static_cast<std::size_t>(packed.size()) };
			dtype = va::DType::Float32;
			return;
		}
		case Variant::PACKED_FLOAT64_ARRAY: {
			const auto packed = PackedFloat64Array(array);
			shape = { static_cast<std::size_t>(packed.size()) };
			dtype = va::DType::Float64;
			return;
		}
		case Variant::PACKED_VECTOR2_ARRAY: {
			const auto packed = PackedVector2Array(array);
			shape = { static_cast<std::size_t>(packed.size()), 2 };
			dtype = va::dtype_of_type<real_t>();
			return;
		}
		case Variant::PACKED_VECTOR3_ARRAY: {
			const auto packed = PackedVector3Array(array);
			shape = { static_cast<std::size_t>(packed.size()), 3 };
			dtype = va::dtype_of_type<real_t>();
			return;
		}
		case Variant::PACKED_VECTOR4_ARRAY: {
			const auto packed = PackedVector4Array(array);
			shape = { static_cast<std::size_t>(packed.size()), 4 };
			dtype = va::dtype_of_type<real_t>();
			return;
		}
		case Variant::PACKED_COLOR_ARRAY: {
			const auto packed = PackedColorArray(array);
			shape = { static_cast<std::size_t>(packed.size()), 4 };
			dtype = va::DType::Float32;
			return;
		}
		case Variant::VECTOR2I: {
			shape = { 2 };
			dtype = va::DType::Int32;
			return;
		}
		case Variant::VECTOR3I: {
			shape = { 3 };
			dtype = va::DType::Int32;
			return;
		}
		case Variant::VECTOR4I: {
			shape = { 4 };
			dtype = va::DType::Int32;
			return;
		}
		case Variant::VECTOR2: {
			shape = { 2 };
			dtype = va::dtype_of_type<real_t>();
			return;
		}
		case Variant::VECTOR3: {
			shape = { 3 };
			dtype = va::dtype_of_type<real_t>();
			return;
		}
		case Variant::VECTOR4: {
			shape = { 4 };
			dtype = va::dtype_of_type<real_t>();
			return;
		}
		case Variant::COLOR: {
			shape = { 4 };
			dtype = va::DType::Float32;
			return;
		}
		default:
			break;
	}

	throw std::runtime_error("Unsupported type");
}

template <typename E, std::size_t... axes, typename T>
auto adapt_packed(const T& packed) {
	return va::util::adapt_c_array(
		const_cast<E*>(reinterpret_cast<const E*>(packed.ptr())),
		{ static_cast<std::size_t>(packed.size()), axes... }
	);
}

template <typename T>
auto adapt_array_tensor(const T& t) {
	const auto& array = numdot::VariantAsArray<T>::get(t);
	using Tensor = numdot::ArrayAsTensor<std::remove_reference_t<decltype(array)>>;

	const auto nat_shape = typename Tensor::shape {};
	va::shape_type shape(Tensor::shape::size());
	std::copy_n(nat_shape.begin(), nat_shape.size(), shape.begin());

	return va::util::adapt_c_array(
		const_cast<std::remove_const_t<typename Tensor::value_type>*>(reinterpret_cast<typename Tensor::value_type*>(&*array)),
		shape
	);
}

std::shared_ptr<va::VArray> array_as_varray(const Array& input_array) {
	va::shape_type shape;
	va::DType dtype = va::DTypeMax;

	find_shape_and_dtype_of_array(shape, dtype, input_array);

	if (dtype == va::DTypeMax) dtype = va::Float64; // Default dtype

	std::shared_ptr<va::VArray> varray = va::empty(va::store::default_allocator, dtype, shape);
	std::vector<std::tuple<xt::xstrided_slice_vector, Array>> next { std::make_tuple(xt::xstrided_slice_vector {}, input_array) };

	while (!next.empty()) {
		const auto [array_base_idx, array] = std::move(next.back());
		next.pop_back();

		for (std::size_t i = 0; i < array.size(); ++i) {
			xt::xstrided_slice_vector element_idx = array_base_idx;
			element_idx.push_back(static_cast<std::ptrdiff_t>(i));

			const auto& array_element = array[i];
			switch (array_element.get_type()) {
				case Variant::OBJECT: {
					if (const auto ndarray = Object::cast_to<NDArray>(array_element)) {
						auto compute = varray->sliced_data(element_idx);
						va::assign(compute, ndarray->array->data);
						continue;
					}
				}
				case Variant::ARRAY:
					next.emplace_back(element_idx, static_cast<Array>(array_element));
					continue;
				case Variant::BOOL: {
					auto compute = varray->sliced_data(element_idx);
					// TODO If we're on the last dimension, we should use element assign rather than slice - fill for all these.
					va::fill(compute, static_cast<bool>(array_element));
					continue;
				}
				case Variant::INT: {
					auto compute = varray->sliced_data(element_idx);
					// TODO If we're on the last dimension, we should use element assign rather than slice - fill for all these.
					va::fill(compute, static_cast<int64_t>(array_element));
					continue;
				}
				case Variant::FLOAT: {
					auto compute = varray->sliced_data(element_idx);
					// TODO If we're on the last dimension, we should use element assign rather than slice - fill for all these.
					va::fill(compute, static_cast<double_t>(array_element));
					continue;
				}
				case Variant::PACKED_BYTE_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedByteArray(array_element);
					va::assign(compute, adapt_packed<uint8_t>(packed));
					continue;
				}
				case Variant::PACKED_INT32_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedInt32Array(array_element);
					va::assign(compute, adapt_packed<int32_t>(packed));
					continue;
				}
				case Variant::PACKED_INT64_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedInt64Array(array_element);
					va::assign(compute, adapt_packed<int64_t>(packed));
					continue;
				}
				case Variant::PACKED_FLOAT32_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedFloat32Array(array_element);
					va::assign(compute, adapt_packed<float_t>(packed));
					continue;
				}
				case Variant::PACKED_FLOAT64_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedFloat64Array(array_element);
					va::assign(compute, adapt_packed<double_t>(packed));
					continue;
				}
				case Variant::PACKED_VECTOR2_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedVector2Array(array_element);
					va::assign(compute, adapt_packed<real_t, 2>(packed));
				}
				case Variant::PACKED_VECTOR3_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedVector3Array(array_element);
					va::assign(compute, adapt_packed<real_t, 3>(packed));
					continue;
				}
				case Variant::PACKED_VECTOR4_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedVector4Array(array_element);
					va::assign(compute, adapt_packed<real_t, 4>(packed));
					continue;
				}
				case Variant::PACKED_COLOR_ARRAY: {
					auto compute = varray->sliced_data(element_idx);
					const auto packed = PackedColorArray(array_element);
					va::assign(compute, adapt_packed<float_t, 4>(packed));
					continue;
				}
				case Variant::VECTOR2I: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Vector2i(array_element)));
					continue;
				}
				case Variant::VECTOR3I: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Vector3i(array_element)));
					continue;
				}
				case Variant::VECTOR4I: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Vector4i(array_element)));
					continue;
				}
				case Variant::VECTOR2: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Vector2(array_element)));
					continue;
				}
				case Variant::VECTOR3: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Vector3(array_element)));
					continue;
				}
				case Variant::VECTOR4: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Vector4(array_element)));
					continue;
				}
				case Variant::COLOR: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Color(array_element)));
					continue;
				}
				case Variant::QUATERNION: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Quaternion(array_element)));
					continue;
				}
				case Variant::PLANE: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Plane(array_element)));
					continue;
				}
				case Variant::BASIS: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Basis(array_element)));
					continue;
				}
				case Variant::PROJECTION: {
					auto compute = varray->sliced_data(element_idx);
					va::assign(compute, adapt_array_tensor(Projection(array_element)));
					continue;
				}
				default:
					break;
			}

			throw std::runtime_error("unsupported array type");
		}
	}

	return varray;
}

std::shared_ptr<va::VArray> variant_as_array(const Variant& array) {
	switch (array.get_type()) {
		case Variant::OBJECT: {
			if (const auto ndarray = Object::cast_to<NDArray>(array)) {
				return ndarray->array;
			}
			break;
		}
		case Variant::ARRAY: {
			return array_as_varray(array);
		}
		case Variant::BOOL: {
			return va::store::from_scalar<bool>(array);
		}
		case Variant::INT: {
			return va::store::from_scalar<int64_t>(array);
		}
		case Variant::FLOAT: {
			return va::store::from_scalar<double_t>(array);
		}
		case Variant::PACKED_BYTE_ARRAY: {
			auto packed = PackedByteArray(array);
			return numdot::varray_from_packed(
			va::util::adapt_c_array(const_cast<uint8_t*>(packed.ptr()), { static_cast<std::size_t>(packed.size()) }),
				std::move(packed)
			);
		}
		case Variant::PACKED_INT32_ARRAY: {
			auto packed = PackedInt32Array(array);
			return numdot::varray_from_packed(
					va::util::adapt_c_array(const_cast<int32_t*>(packed.ptr()), { static_cast<std::size_t>(packed.size()) }),
				std::move(packed)
			);
		}
		case Variant::PACKED_INT64_ARRAY: {
			auto packed = PackedInt64Array(array);
			return numdot::varray_from_packed(
			va::util::adapt_c_array(const_cast<int64_t*>(packed.ptr()), { static_cast<std::size_t>(packed.size()) }),
				std::move(packed)
			);
		}
		case Variant::PACKED_FLOAT32_ARRAY: {
			auto packed = PackedFloat32Array(array);
			return numdot::varray_from_packed(
			va::util::adapt_c_array(const_cast<float_t*>(packed.ptr()), { static_cast<std::size_t>(packed.size()) }),
				std::move(packed)
			);
		}
		case Variant::PACKED_FLOAT64_ARRAY: {
			auto packed = PackedFloat64Array(array);
			return numdot::varray_from_packed(
			va::util::adapt_c_array(const_cast<double_t*>(packed.ptr()), { static_cast<std::size_t>(packed.size()) }),
				std::move(packed)
			);
		}
		case Variant::PACKED_VECTOR2_ARRAY: {
			auto packed = PackedVector2Array(array);
			return numdot::varray_from_packed(
			va::util::adapt_c_array(const_cast<real_t*>(&packed.ptr()[0].coord[0]), { static_cast<std::size_t>(packed.size()), 2 }),
				std::move(packed)
			);
		}
		case Variant::PACKED_VECTOR3_ARRAY: {
			auto packed = PackedVector3Array(array);
			return numdot::varray_from_packed(
			va::util::adapt_c_array(const_cast<real_t*>(&packed.ptr()[0].coord[0]), { static_cast<std::size_t>(packed.size()), 3 }),
				std::move(packed)
			);
		}
		case Variant::PACKED_VECTOR4_ARRAY: {
			auto packed = PackedVector4Array(array);
			return numdot::varray_from_packed(
			va::util::adapt_c_array(const_cast<real_t*>(&packed.ptr()[0].coord[0]), { static_cast<std::size_t>(packed.size()), 4 }),
				std::move(packed)
			);
		}
		case Variant::PACKED_COLOR_ARRAY: {
			auto packed = PackedColorArray(array);
			return numdot::varray_from_packed(
			va::util::adapt_c_array(const_cast<float_t*>(&packed.ptr()[0].components[0]), { static_cast<std::size_t>(packed.size()), 4 }),
				std::move(packed)
			);
		}
		case Variant::VECTOR2I: {
			return numdot::varray_from_tensor(Vector2i(array));
		}
		case Variant::VECTOR3I: {
			return numdot::varray_from_tensor(Vector3i(array));
		}
		case Variant::VECTOR4I: {
			return numdot::varray_from_tensor(Vector4i(array));
		}
		case Variant::VECTOR2: {
			return numdot::varray_from_tensor(Vector2(array));
		}
		case Variant::VECTOR3: {
			return numdot::varray_from_tensor(Vector3(array));
		}
		case Variant::VECTOR4: {
			return numdot::varray_from_tensor(Vector4(array));
		}
		case Variant::COLOR: {
			return numdot::varray_from_tensor(Color(array));
		}
		case Variant::QUATERNION: {
			return numdot::varray_from_tensor(Quaternion(array));
		}
		case Variant::BASIS: {
			return numdot::varray_from_tensor(Basis(array));
		}
		case Variant::PROJECTION: {
			return numdot::varray_from_tensor(Projection(array));
		}
		case Variant::PLANE: {
			return numdot::varray_from_tensor(Plane(array));
		}
		default:
			break;
	}

	throw std::runtime_error("Unsupported type");
}

std::shared_ptr<va::VArray> ndarray_as_dtype(const NDArray& ndarray, const va::DType dtype) {
	if (dtype == va::DTypeMax || ndarray.array->dtype() == dtype)
		return ndarray.array;

	return va::copy_as_dtype(va::store::default_allocator, ndarray.array->data, dtype);
}

std::shared_ptr<va::VArray> variant_as_array(const Variant& array, const va::DType dtype, const bool copy) {
	switch (array.get_type()) {
		case Variant::OBJECT: {
			if (const auto ndarray = Object::cast_to<NDArray>(array)) {
				if (copy)
					return va::copy_as_dtype(va::store::default_allocator, ndarray->array->data, dtype);

				return ndarray_as_dtype(*ndarray, dtype);
			}
		}
		default:
			break;
	}

	// Guaranteed to be a fresh array, because we handled non-fresh cases already.
	auto varray = variant_as_array(array);
	if (dtype == va::DTypeMax || dtype == varray->dtype())
		return varray;
	else
		return va::copy_as_dtype(va::store::default_allocator, varray->data, dtype);
}

std::vector<std::shared_ptr<va::VArray>> variant_to_vector(const Variant& array) {
	switch (array.get_type()) {
		case Variant::ARRAY: {
			const Array gdarray = array;

			const std::size_t outer_dim_size = gdarray.size();
			std::vector<std::shared_ptr<va::VArray>> vector(outer_dim_size);
			for (std::size_t i = 0; i < outer_dim_size; i++) {
				xt::xstrided_slice_vector idx {i};
				vector[i] = variant_as_array(gdarray[static_cast<int64_t>(i)]);
			}
			return vector;
		}
		default:
			break;
	}

	const auto ndarray = variant_as_array(array);
	if (ndarray->dimension() < 1) throw std::runtime_error("Invalid array dimension");

	const std::size_t outer_dim_size = ndarray->shape()[0];
	std::vector<std::shared_ptr<va::VArray>> vector(outer_dim_size);
	for (std::size_t i = 0; i < outer_dim_size; i++) {
		xt::xstrided_slice_vector idx {i};
		vector[i] = ndarray->sliced(idx);
	}
	return vector;
}
