#include "conversion_array.hpp"

#include <vatensor/allocate.hpp>                         // for copy_as_dtype
#include <vatensor/vassign.hpp>                          // for assign
#include <cmath>                                       // for double_t, float_t
#include <cstddef>                                     // for size_t
#include <cstdint>                                     // for int64_t, int32_t
#include <iterator>                                    // for size
#include <stdexcept>                                   // for runtime_error
#include <tuple>                                       // for tuple, make_tuple
#include <utility>                                     // for move
#include <vector>                                      // for allocator, vector
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
#include "xtensor/xarray.hpp"                          // for xarray_adaptor
#include "xtensor/xbuffer_adaptor.hpp"                 // for xbuffer_adaptor
#include "xtensor/xlayout.hpp"                         // for layout_type
#include "xtensor/xstorage.hpp"                        // for svector, uvector
#include "xtensor/xstrided_view.hpp"                   // for xstrided_slice...

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
							auto varray_dim_idx = current_dim_idx;
							for (const auto size : ndarray->array->shape()) {
								add_size_at_idx(shape, varray_dim_idx, size);
								varray_dim_idx++;
							}
							continue;
						}
					}
					case Variant::ARRAY:
						next_dim_arrays.push_back(array_element);
						continue;
					case Variant::BOOL:
						dtype = va::dtype_common_type(dtype, va::Bool);
						continue;
					case Variant::INT:
						dtype = va::dtype_common_type(dtype, va::Int64);
						continue;
					case Variant::FLOAT:
						dtype = va::dtype_common_type(dtype, va::Float64);
						continue;
					case Variant::PACKED_BYTE_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(uint8_t()));
						PackedByteArray packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_INT32_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(int32_t()));
						PackedInt32Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_INT64_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(int64_t()));
						PackedInt64Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_FLOAT32_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(float_t()));
						PackedFloat32Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_FLOAT64_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(double_t()));
						PackedFloat64Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						continue;
					}
					case Variant::PACKED_VECTOR2_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(real_t()));
						PackedVector2Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						add_size_at_idx(shape, current_dim_idx + 2, 2);
						continue;
					}
					case Variant::PACKED_VECTOR3_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(real_t()));
						PackedVector3Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						add_size_at_idx(shape, current_dim_idx + 2, 3);
						continue;
					}
					case Variant::PACKED_VECTOR4_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(real_t()));
						PackedVector4Array packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						add_size_at_idx(shape, current_dim_idx + 2, 4);
						continue;
					}
					case Variant::PACKED_COLOR_ARRAY: {
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(float_t()));
						PackedColorArray packed = array_element;
						add_size_at_idx(shape, current_dim_idx + 1, packed.size());
						add_size_at_idx(shape, current_dim_idx + 2, 4);
						continue;
					}
					case Variant::VECTOR2I:
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(int64_t()));
						add_size_at_idx(shape, current_dim_idx + 1, 2);
						continue;
					case Variant::VECTOR3I:
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(int64_t()));
						add_size_at_idx(shape, current_dim_idx + 1, 3);
						continue;
					case Variant::VECTOR4I:
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(int64_t()));
						add_size_at_idx(shape, current_dim_idx + 1, 4);
						continue;
					case Variant::VECTOR2:
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(real_t()));
						add_size_at_idx(shape, current_dim_idx + 1, 2);
						continue;
					case Variant::VECTOR3:
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(real_t()));
						add_size_at_idx(shape, current_dim_idx + 1, 3);
						continue;
					case Variant::VECTOR4:
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(real_t()));
						add_size_at_idx(shape, current_dim_idx + 1, 4);
						continue;
					case Variant::COLOR:
						dtype = va::dtype_common_type(dtype, va::variant_to_dtype(float_t()));
						add_size_at_idx(shape, current_dim_idx + 1, 4);
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
			shape = { static_cast<size_t>(packed.size()) };
			dtype = va::DType::UInt8;
			return;
		}
		case Variant::PACKED_INT32_ARRAY: {
			const auto packed = PackedInt32Array(array);
			shape = { static_cast<size_t>(packed.size()) };
			dtype = va::DType::Int32;
			return;
		}
		case Variant::PACKED_INT64_ARRAY: {
			const auto packed = PackedInt64Array(array);
			shape = { static_cast<size_t>(packed.size()) };
			dtype = va::DType::Int64;
			return;
		}
		case Variant::PACKED_FLOAT32_ARRAY: {
			const auto packed = PackedFloat32Array(array);
			shape = { static_cast<size_t>(packed.size()) };
			dtype = va::DType::Float32;
			return;
		}
		case Variant::PACKED_FLOAT64_ARRAY: {
			const auto packed = PackedFloat64Array(array);
			shape = { static_cast<size_t>(packed.size()) };
			dtype = va::DType::Float64;
			return;
		}
		case Variant::PACKED_VECTOR2_ARRAY: {
			const auto packed = PackedVector2Array(array);
			shape = { static_cast<size_t>(packed.size()), 2 };
			dtype = va::variant_to_dtype(real_t());
			return;
		}
		case Variant::PACKED_VECTOR3_ARRAY: {
			const auto packed = PackedVector3Array(array);
			shape = { static_cast<size_t>(packed.size()), 3 };
			dtype = va::variant_to_dtype(real_t());
			return;
		}
		case Variant::PACKED_VECTOR4_ARRAY: {
			const auto packed = PackedVector4Array(array);
			shape = { static_cast<size_t>(packed.size()), 4 };
			dtype = va::variant_to_dtype(real_t());
			return;
		}
		case Variant::PACKED_COLOR_ARRAY: {
			const auto packed = PackedColorArray(array);
			shape = { static_cast<size_t>(packed.size()), 4 };
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
			dtype = va::variant_to_dtype(real_t());
			return;
		}
		case Variant::VECTOR3: {
			shape = { 3 };
			dtype = va::variant_to_dtype(real_t());
			return;
		}
		case Variant::VECTOR4: {
			shape = { 4 };
			dtype = va::variant_to_dtype(real_t());
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

std::shared_ptr<va::VArray> array_as_varray(const Array& input_array) {
	va::shape_type shape;
	va::DType dtype = va::DTypeMax;

	find_shape_and_dtype_of_array(shape, dtype, input_array);

	if (dtype == va::DTypeMax) dtype = va::Float64; // Default dtype

	std::shared_ptr<va::VArray> varray = va::empty(dtype, shape);
	std::vector<std::tuple<xt::xstrided_slice_vector, Array>> next = { std::make_tuple(xt::xstrided_slice_vector {}, input_array) };

	while (!next.empty()) {
		const auto [array_base_idx, array] = std::move(next.back());
		next.pop_back();

		for (std::size_t i = 0; i < array.size(); ++i) {
			auto element_idx = array_base_idx;
			element_idx.emplace_back(i);

			const auto& array_element = array[i];
			switch (array_element.get_type()) {
				case Variant::OBJECT: {
					if (const auto ndarray = Object::cast_to<NDArray>(array_element)) {
						auto compute = varray->sliced_write(element_idx);
						va::assign(compute, ndarray->array->read);
						continue;
					}
				}
				case Variant::ARRAY:
					next.emplace_back(element_idx, static_cast<Array>(array_element));
					continue;
				case Variant::BOOL: {
					auto compute = varray->sliced_write(element_idx);
					// TODO If we're on the last dimension, we should use element assign rather than slice - fill for all these.
					va::assign(compute, static_cast<bool>(array_element));
					continue;
				}
				case Variant::INT: {
					auto compute = varray->sliced_write(element_idx);
					// TODO If we're on the last dimension, we should use element assign rather than slice - fill for all these.
					va::assign(compute, static_cast<int64_t>(array_element));
					continue;
				}
				case Variant::FLOAT: {
					auto compute = varray->sliced_write(element_idx);
					// TODO If we're on the last dimension, we should use element assign rather than slice - fill for all these.
					va::assign(compute, static_cast<double_t>(array_element));
					continue;
				}
				case Variant::PACKED_BYTE_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedByteArray(array_element);
					va::assign(
						compute,
						adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
					);
					continue;
				}
				case Variant::PACKED_INT32_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedInt32Array(array_element);
					va::assign(
						compute,
						adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
					);
					continue;
				}
				case Variant::PACKED_INT64_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedInt64Array(array_element);
					va::assign(
						compute,
						adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
					);
					continue;
				}
				case Variant::PACKED_FLOAT32_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedFloat32Array(array_element);
					va::assign(
						compute,
						adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
					);
					continue;
				}
				case Variant::PACKED_FLOAT64_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedFloat64Array(array_element);
					va::assign(
						compute,
						adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
					);
					continue;
				}
				case Variant::PACKED_VECTOR2_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedVector2Array(array_element);
					va::assign(
						compute,
						adapt_c_array(&packed.ptr()[0].coord[0], { static_cast<std::size_t>(packed.size()), 2 })
					);
					continue;
				}
				case Variant::PACKED_VECTOR3_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedVector3Array(array_element);
					va::assign(
						compute,
						adapt_c_array(&packed.ptr()[0].coord[0], { static_cast<std::size_t>(packed.size()), 3 })
					);
					continue;
				}
				case Variant::PACKED_VECTOR4_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedVector4Array(array_element);
					va::assign(
						compute,
						adapt_c_array(&packed.ptr()[0].components[0], { static_cast<std::size_t>(packed.size()), 4 })
					);
					continue;
				}
				case Variant::PACKED_COLOR_ARRAY: {
					auto compute = varray->sliced_write(element_idx);
					const auto packed = PackedColorArray(array_element);
					va::assign(
						compute,
						adapt_c_array(&packed.ptr()[0].components[0], { static_cast<std::size_t>(packed.size()), 4 })
					);
					continue;
				}
				case Variant::VECTOR2I: {
					auto compute = varray->sliced_write(element_idx);
					const Vector2i vector = array_element;
					va::assign(
						compute,
						adapt_c_array(&vector.coord[0], { std::size(vector.coord) })
					);
					continue;
				}
				case Variant::VECTOR3I: {
					auto compute = varray->sliced_write(element_idx);
					const Vector3i vector = array_element;
					va::assign(
						compute,
						adapt_c_array(&vector.coord[0], { std::size(vector.coord) })
					);
					continue;
				}
				case Variant::VECTOR4I: {
					auto compute = varray->sliced_write(element_idx);
					const Vector4i vector = array_element;
					va::assign(
						compute,
						adapt_c_array(&vector.coord[0], { std::size(vector.coord) })
					);
					continue;
				}
				case Variant::VECTOR2: {
					auto compute = varray->sliced_write(element_idx);
					const Vector2 vector = array_element;
					va::assign(
						compute,
						adapt_c_array(&vector.coord[0], { std::size(vector.coord) })
					);
					continue;
				}
				case Variant::VECTOR3: {
					auto compute = varray->sliced_write(element_idx);
					const Vector3 vector = array_element;
					va::assign(
						compute,
						adapt_c_array(&vector.coord[0], { std::size(vector.coord) })
					);
					continue;
				}
				case Variant::VECTOR4: {
					auto compute = varray->sliced_write(element_idx);
					const Vector4 vector = array_element;
					va::assign(
						compute,
						adapt_c_array(&vector.components[0], { std::size(vector.components) })
					);
					continue;
				}
				case Variant::COLOR: {
					auto compute = varray->sliced_write(element_idx);
					const Color vector = array_element;
					va::assign(
						compute,
						adapt_c_array(&vector.components[0], { std::size(vector.components) })
					);
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
			return va::from_scalar<bool>(array);
		}
		case Variant::INT: {
			return va::from_scalar<int64_t>(array);
		}
		case Variant::FLOAT: {
			return va::from_scalar<double_t>(array);
		}
		case Variant::PACKED_BYTE_ARRAY: {
			const auto packed = PackedByteArray(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
				)
			);
		}
		case Variant::PACKED_INT32_ARRAY: {
			const auto packed = PackedInt32Array(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
				)
			);
		}
		case Variant::PACKED_INT64_ARRAY: {
			const auto packed = PackedInt64Array(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
				)
			);
		}
		case Variant::PACKED_FLOAT32_ARRAY: {
			const auto packed = PackedFloat32Array(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
				)
			);
		}
		case Variant::PACKED_FLOAT64_ARRAY: {
			const auto packed = PackedFloat64Array(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(packed.ptr(), { static_cast<std::size_t>(packed.size()) })
				)
			);
		}
		case Variant::PACKED_VECTOR2_ARRAY: {
			const auto packed = PackedVector2Array(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(&packed.ptr()[0].coord[0], { static_cast<std::size_t>(packed.size()), 2 })
				)
			);
		}
		case Variant::PACKED_VECTOR3_ARRAY: {
			const auto packed = PackedVector3Array(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(&packed.ptr()[0].coord[0], { static_cast<std::size_t>(packed.size()), 3 })
				)
			);
		}
		case Variant::PACKED_VECTOR4_ARRAY: {
			const auto packed = PackedVector4Array(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(&packed.ptr()[0].components[0], { static_cast<std::size_t>(packed.size()), 4 })
				)
			);
		}
		case Variant::PACKED_COLOR_ARRAY: {
			const auto packed = PackedColorArray(array);
			return va::from_store(
				va::make_store(
					adapt_c_array(&packed.ptr()[0].components[0], { static_cast<std::size_t>(packed.size()), 4 })
				)
			);
		}
		case Variant::VECTOR2I: {
			auto vector = Vector2i(array);
			return va::from_store(
				va::make_store(
					{ vector.x, vector.y }
				)
			);
		}
		case Variant::VECTOR3I: {
			auto vector = Vector3i(array);
			return va::from_store(
				va::make_store(
					{ vector.x, vector.y, vector.z }
				)
			);
		}
		case Variant::VECTOR4I: {
			auto vector = Vector4i(array);
			return va::from_store(
				va::make_store(
					{ vector.x, vector.y, vector.z, vector.w }
				)
			);
		}
		case Variant::VECTOR2: {
			auto vector = Vector2(array);
			return va::from_store(
				va::make_store(
					{ vector.x, vector.y }
				)
			);
		}
		case Variant::VECTOR3: {
			auto vector = Vector3(array);
			return va::from_store(
				va::make_store(
					{ vector.x, vector.y, vector.z }
				)
			);
		}
		case Variant::VECTOR4: {
			auto vector = Vector4(array);
			return va::from_store(
				va::make_store(
					{ vector.x, vector.y, vector.z, vector.w }
				)
			);
		}
		case Variant::COLOR: {
			auto vector = Color(array);
			return va::from_store(
				va::make_store(
					{ vector.r, vector.g, vector.b, vector.a }
				)
			);
		}
		default:
			break;
	}

	throw std::runtime_error("Unsupported type");
}

std::shared_ptr<va::VArray> variant_as_array(const Variant& array, const va::DType dtype, const bool copy) {
	switch (array.get_type()) {
		case Variant::OBJECT: {
			if (const auto ndarray = Object::cast_to<NDArray>(array)) {
				if (!copy && (dtype == va::DTypeMax || ndarray->array->dtype() == dtype))
					return ndarray->array;
				else
					return va::copy_as_dtype(*ndarray->array, dtype);
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
		return va::copy_as_dtype(*varray, dtype);
}
