#include "ndarray.hpp"

#include <gdconvert/conversion_ints.hpp>             // for variants_to_axes
#include <vatensor/comparison.hpp>                   // for equal_to, greater
#include <vatensor/linalg.hpp>                       // for reduce_dot, dot
#include <vatensor/logical.hpp>                      // for logical_and, logic...
#include <vatensor/bitwise.hpp>                      // for bitwise_and, bitwise...
#include <vatensor/reduce.hpp>                       // for all, any, max, mean
#include <vatensor/trigonometry.hpp>                 // for acos, acosh, asin
#include <vatensor/vassign.hpp>                      // for assign
#include <vatensor/vmath.hpp>                        // for abs, add, clip
#include <vatensor/xtensor_store.hpp>                        // for abs, add, clip
#include <vatensor/vcarray.hpp>                        // fill_c_array_flat
#include <algorithm>                               // for copy
#include <cstddef>                                 // for size_t
#include <ndutil.hpp>
#include <stdexcept>                               // for runtime_error
#include <variant>                                 // for visit
#include <gdconvert/packed_array_store.hpp>
#include <vatensor/create.hpp>
#include <vatensor/dtype.hpp>
#include <vatensor/rearrange.hpp>
#include "gdconvert/conversion_array.hpp"            // for fill_c_array_flat
#include "gdconvert/conversion_slice.hpp"            // for variants_to_slice_...
#include "gdconvert/conversion_string.hpp"           // for xt_to_string
#include "gdconvert/conversion_scalar.hpp"
#include "gdconvert/variant_tensor.hpp"
#include "godot_cpp/classes/global_constants.hpp"  // for MethodFlags
#include "godot_cpp/core/class_db.hpp"             // for D_METHOD, ClassDB
#include "godot_cpp/core/error_macros.hpp"         // for ERR_FAIL_COND_V_MSG
#include "godot_cpp/core/memory.hpp"               // for _post_initialize
#include "godot_cpp/variant/string_name.hpp"       // for StringName
#include "godot_cpp/variant/variant.hpp"           // for Variant
#include "nd.hpp"                                    // for nd
#include "vatensor/round.hpp"                        // for ceil, floor, nearb...
#include "vatensor/varray.hpp"                       // for VArray, VArrayTarget
#include "xtensor/xiterator.hpp"                   // for operator==
#include "xtensor/xstrided_view.hpp"               // for xstrided_slice_vector
#include "xtl/xiterator_base.hpp"                  // for operator!=
#include "vatensor/stride_tricks.hpp"

using namespace godot;

void NDArray::_bind_methods() {
	BIND_ENUM_CONSTANT(RowMajor);
	BIND_ENUM_CONSTANT(ColumnMajor);
	BIND_ENUM_CONSTANT(Dynamic);
	BIND_ENUM_CONSTANT(Any);

	godot::ClassDB::bind_method(D_METHOD("dtype"), &NDArray::dtype);
	godot::ClassDB::bind_method(D_METHOD("shape"), &NDArray::shape);
	godot::ClassDB::bind_method(D_METHOD("size"), &NDArray::size);
	godot::ClassDB::bind_method(D_METHOD("buffer_dtype"), &NDArray::buffer_dtype);
	godot::ClassDB::bind_method(D_METHOD("buffer_size"), &NDArray::buffer_size);
	godot::ClassDB::bind_method(D_METHOD("buffer_size_in_bytes"), &NDArray::buffer_size_in_bytes);
	godot::ClassDB::bind_method(D_METHOD("ndim"), &NDArray::ndim);
	godot::ClassDB::bind_method(D_METHOD("strides"), &NDArray::strides);
	godot::ClassDB::bind_method(D_METHOD("strides_layout"), &NDArray::strides_layout);
	godot::ClassDB::bind_method(D_METHOD("strides_offset"), &NDArray::strides_offset);

	ClassDB::bind_method(D_METHOD("_iter_init"), &NDArray::_iter_init);
	ClassDB::bind_method(D_METHOD("_iter_get"), &NDArray::_iter_get);
	ClassDB::bind_method(D_METHOD("_iter_next"), &NDArray::_iter_next);

	numdot::bind_vararg_method(numdot::VD_METHOD("set", PropertyInfo(Variant::NIL, "value")), &NDArray::set);
	numdot::bind_vararg_method(numdot::VD_METHOD("get"), &NDArray::get);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_bool"), &NDArray::get_bool);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_int"), &NDArray::get_int);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_float"), &NDArray::get_float);

	godot::ClassDB::bind_method(D_METHOD("as_type", "type"), &NDArray::as_type);
	godot::ClassDB::bind_method(D_METHOD("copy"), &NDArray::copy);
	numdot::bind_vararg_method(numdot::VD_METHOD("transpose"), &NDArray::transpose);
	godot::ClassDB::bind_method(D_METHOD("flatten"), &NDArray::flatten);

	numdot::bind_vararg_method(numdot::VD_METHOD("get_vector2"), &NDArray::get_vector2);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_vector3"), &NDArray::get_vector3);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_vector4"), &NDArray::get_vector4);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_vector2i"), &NDArray::get_vector2i);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_vector3i"), &NDArray::get_vector3i);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_vector4i"), &NDArray::get_vector4i);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_color"), &NDArray::get_color);

	numdot::bind_vararg_method(numdot::VD_METHOD("get_quaternion"), &NDArray::get_quaternion);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_plane"), &NDArray::get_plane);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_basis"), &NDArray::get_basis);
	numdot::bind_vararg_method(numdot::VD_METHOD("get_projection"), &NDArray::get_projection);

	godot::ClassDB::bind_method(D_METHOD("to_bool"), &NDArray::to_bool);
	godot::ClassDB::bind_method(D_METHOD("to_int"), &NDArray::to_int);
	godot::ClassDB::bind_method(D_METHOD("to_float"), &NDArray::to_float);

	godot::ClassDB::bind_method(D_METHOD("to_vector2"), &NDArray::to_vector2);
	godot::ClassDB::bind_method(D_METHOD("to_vector3"), &NDArray::to_vector3);
	godot::ClassDB::bind_method(D_METHOD("to_vector4"), &NDArray::to_vector4);
	godot::ClassDB::bind_method(D_METHOD("to_vector2i"), &NDArray::to_vector2i);
	godot::ClassDB::bind_method(D_METHOD("to_vector3i"), &NDArray::to_vector3i);
	godot::ClassDB::bind_method(D_METHOD("to_vector4i"), &NDArray::to_vector4i);
	godot::ClassDB::bind_method(D_METHOD("to_color"), &NDArray::to_color);

	godot::ClassDB::bind_method(D_METHOD("to_quaternion"), &NDArray::to_quaternion);
	godot::ClassDB::bind_method(D_METHOD("to_plane"), &NDArray::to_plane);
	godot::ClassDB::bind_method(D_METHOD("to_basis"), &NDArray::to_basis);
	godot::ClassDB::bind_method(D_METHOD("to_projection"), &NDArray::to_projection);

	godot::ClassDB::bind_method(D_METHOD("to_packed_float32_array"), &NDArray::to_packed_float32_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_float64_array"), &NDArray::to_packed_float64_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_byte_array"), &NDArray::to_packed_byte_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_int32_array"), &NDArray::to_packed_int32_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_int64_array"), &NDArray::to_packed_int64_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_vector2_array"), &NDArray::to_packed_vector2_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_vector3_array"), &NDArray::to_packed_vector3_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_vector4_array"), &NDArray::to_packed_vector4_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_color_array"), &NDArray::to_packed_color_array);

	godot::ClassDB::bind_method(D_METHOD("to_godot_array"), &NDArray::to_godot_array);

	godot::ClassDB::bind_method(D_METHOD("assign_conjugate", "a"), &NDArray::assign_conjugate);
	godot::ClassDB::bind_method(D_METHOD("assign_angle", "a"), &NDArray::assign_angle);

	godot::ClassDB::bind_method(D_METHOD("assign_positive", "a"), &NDArray::assign_positive);
	godot::ClassDB::bind_method(D_METHOD("assign_negative", "a"), &NDArray::assign_negative);
	godot::ClassDB::bind_method(D_METHOD("assign_add", "a", "b"), &NDArray::assign_add);
	godot::ClassDB::bind_method(D_METHOD("assign_subtract", "a", "b"), &NDArray::assign_subtract);
	godot::ClassDB::bind_method(D_METHOD("assign_multiply", "a", "b"), &NDArray::assign_multiply);
	godot::ClassDB::bind_method(D_METHOD("assign_divide", "a", "b"), &NDArray::assign_divide);
	godot::ClassDB::bind_method(D_METHOD("assign_remainder", "a", "b"), &NDArray::assign_remainder);
	godot::ClassDB::bind_method(D_METHOD("assign_pow", "a", "b"), &NDArray::assign_pow);

	godot::ClassDB::bind_method(D_METHOD("assign_minimum", "a", "b"), &NDArray::assign_minimum);
	godot::ClassDB::bind_method(D_METHOD("assign_maximum", "a", "b"), &NDArray::assign_maximum);
	godot::ClassDB::bind_method(D_METHOD("assign_clip", "a", "min", "max"), &NDArray::assign_clip);

	godot::ClassDB::bind_method(D_METHOD("assign_sign", "a"), &NDArray::assign_sign);
	godot::ClassDB::bind_method(D_METHOD("assign_abs", "a"), &NDArray::assign_abs);
	godot::ClassDB::bind_method(D_METHOD("assign_square", "a"), &NDArray::assign_square);
	godot::ClassDB::bind_method(D_METHOD("assign_sqrt", "a"), &NDArray::assign_sqrt);

	godot::ClassDB::bind_method(D_METHOD("assign_exp", "a"), &NDArray::assign_exp);
	godot::ClassDB::bind_method(D_METHOD("assign_log", "a"), &NDArray::assign_log);

	godot::ClassDB::bind_method(D_METHOD("assign_rad2deg", "a"), &NDArray::assign_rad2deg);
	godot::ClassDB::bind_method(D_METHOD("assign_deg2rad", "a"), &NDArray::assign_deg2rad);

	godot::ClassDB::bind_method(D_METHOD("assign_sin", "a"), &NDArray::assign_sin);
	godot::ClassDB::bind_method(D_METHOD("assign_cos", "a"), &NDArray::assign_cos);
	godot::ClassDB::bind_method(D_METHOD("assign_tan", "a"), &NDArray::assign_tan);
	godot::ClassDB::bind_method(D_METHOD("assign_asin", "a"), &NDArray::assign_asin);
	godot::ClassDB::bind_method(D_METHOD("assign_acos", "a"), &NDArray::assign_acos);
	godot::ClassDB::bind_method(D_METHOD("assign_atan", "a"), &NDArray::assign_atan);
	godot::ClassDB::bind_method(D_METHOD("assign_atan2", "x1", "x2"), &NDArray::assign_atan2);

	godot::ClassDB::bind_method(D_METHOD("assign_sinh", "a"), &NDArray::assign_sinh);
	godot::ClassDB::bind_method(D_METHOD("assign_cosh", "a"), &NDArray::assign_cosh);
	godot::ClassDB::bind_method(D_METHOD("assign_tanh", "a"), &NDArray::assign_tanh);
	godot::ClassDB::bind_method(D_METHOD("assign_asinh", "a"), &NDArray::assign_asinh);
	godot::ClassDB::bind_method(D_METHOD("assign_acosh", "a"), &NDArray::assign_acosh);
	godot::ClassDB::bind_method(D_METHOD("assign_atanh", "a"), &NDArray::assign_atanh);

	godot::ClassDB::bind_method(D_METHOD("assign_sum", "a", "axes"), &NDArray::assign_sum, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_prod", "a", "axes"), &NDArray::assign_prod, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_mean", "a", "axes"), &NDArray::assign_mean, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_var", "a", "axes"), &NDArray::assign_variance, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_std", "a", "axes"), &NDArray::assign_standard_deviation, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_max", "a", "axes"), &NDArray::assign_max, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_min", "a", "axes"), &NDArray::assign_min, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_norm", "a", "ord", "axes"), &NDArray::assign_norm, DEFVAL(2), DEFVAL(nullptr));

	godot::ClassDB::bind_method(D_METHOD("assign_floor", "a"), &NDArray::assign_floor);
	godot::ClassDB::bind_method(D_METHOD("assign_ceil", "a"), &NDArray::assign_ceil);
	godot::ClassDB::bind_method(D_METHOD("assign_round", "a"), &NDArray::assign_round);
	godot::ClassDB::bind_method(D_METHOD("assign_trunc", "a"), &NDArray::assign_trunc);
	godot::ClassDB::bind_method(D_METHOD("assign_rint", "a"), &NDArray::assign_rint);

	godot::ClassDB::bind_method(D_METHOD("assign_equal", "a", "b"), &NDArray::assign_equal);
	godot::ClassDB::bind_method(D_METHOD("assign_not_equal", "a", "b"), &NDArray::assign_not_equal);
	godot::ClassDB::bind_method(D_METHOD("assign_greater", "a", "b"), &NDArray::assign_greater);
	godot::ClassDB::bind_method(D_METHOD("assign_greater_equal", "a", "b"), &NDArray::assign_greater_equal);
	godot::ClassDB::bind_method(D_METHOD("assign_less", "a", "b"), &NDArray::assign_less);
	godot::ClassDB::bind_method(D_METHOD("assign_less_equal", "a", "b"), &NDArray::assign_less_equal);

	godot::ClassDB::bind_method(D_METHOD("assign_logical_and", "a", "b"), &NDArray::assign_logical_and);
	godot::ClassDB::bind_method(D_METHOD("assign_logical_or", "a", "b"), &NDArray::assign_logical_or);
	godot::ClassDB::bind_method(D_METHOD("assign_logical_xor", "a", "b"), &NDArray::assign_logical_xor);
	godot::ClassDB::bind_method(D_METHOD("assign_logical_not", "a"), &NDArray::assign_logical_not);
	godot::ClassDB::bind_method(D_METHOD("assign_all", "a", "axes"), &NDArray::assign_all, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_any", "a", "axes"), &NDArray::assign_any, DEFVAL(nullptr));

	godot::ClassDB::bind_method(D_METHOD("assign_bitwise_and", "a", "b"), &NDArray::assign_bitwise_and);
	godot::ClassDB::bind_method(D_METHOD("assign_bitwise_or", "a", "b"), &NDArray::assign_bitwise_or);
	godot::ClassDB::bind_method(D_METHOD("assign_bitwise_xor", "a", "b"), &NDArray::assign_bitwise_xor);
	godot::ClassDB::bind_method(D_METHOD("assign_bitwise_not", "a"), &NDArray::assign_bitwise_not);
	godot::ClassDB::bind_method(D_METHOD("assign_bitwise_left_shift", "a", "b"), &NDArray::assign_bitwise_left_shift);
	godot::ClassDB::bind_method(D_METHOD("assign_bitwise_right_shift", "a", "b"), &NDArray::assign_bitwise_right_shift);

	godot::ClassDB::bind_method(D_METHOD("assign_dot", "a", "b"), &NDArray::assign_dot);
	godot::ClassDB::bind_method(D_METHOD("assign_reduce_dot", "a", "b", "axes"), &NDArray::assign_reduce_dot, DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_matmul", "a", "b"), &NDArray::assign_matmul);
	godot::ClassDB::bind_method(D_METHOD("assign_cross", "a", "b", "axisa", "axisb", "axisc"), &NDArray::assign_cross, DEFVAL(-1), DEFVAL(-1), DEFVAL(-1));

	godot::ClassDB::bind_method(D_METHOD("assign_convolve", "array", "kernel"), &NDArray::assign_convolve);
}

NDArray::NDArray() = default;

NDArray::~NDArray() = default;

String NDArray::_to_string() const {
	return std::visit([](auto&& arg) { return xt_to_string(arg); }, array->data);
}

va::DType NDArray::dtype() const {
	return array->dtype();
}

PackedInt64Array NDArray::shape() const {
	PackedInt64Array packed;
	const auto& shape = array->shape();
	packed.resize(static_cast<int64_t>(shape.size()));
	std::copy_n(shape.begin(), shape.size(), packed.ptrw());
	return packed;
}

uint64_t NDArray::size() const {
	return array->size();
}

uint64_t NDArray::buffer_dtype() const {
	return array->store->dtype();
}

uint64_t NDArray::buffer_size() const {
	return array->store->size();
}

uint64_t NDArray::buffer_size_in_bytes() const {
	return array->store->size() * va::size_of_dtype_in_bytes_unchecked(array->store->dtype());
}

uint64_t NDArray::ndim() const {
	return array->dimension();
}

PackedInt64Array NDArray::strides() const {
	PackedInt64Array packed;
	const auto& strides = array->strides();
	packed.resize(static_cast<int64_t>(strides.size()));
	std::copy_n(strides.begin(), strides.size(), packed.ptrw());
	return packed;
}

NDArray::Layout NDArray::strides_layout() const {
	switch (array->layout()) {
		case xt::layout_type::row_major:
			return NDArray::Layout::RowMajor;
		case xt::layout_type::column_major:
			return NDArray::Layout::ColumnMajor;
		case xt::layout_type::dynamic:
			return NDArray::Layout::Dynamic;
		case xt::layout_type::any:
			return NDArray::Layout::Any;
		default: ;
			ERR_FAIL_V_MSG(NDArray::Dynamic, "Unrecognized layout found.");
	}
}

uint64_t NDArray::strides_offset() const {
	return static_cast<uint64_t>(array->data_offset);
}

Variant NDArray::_iter_init(const Array& p_iter) {
	ERR_FAIL_COND_V_MSG(array->shape().empty(), false, "iteration over a 0-d array");

	Array ref = p_iter;
	ERR_FAIL_COND_V_MSG(ref.size() != 1, false, "size of iterator cache must be 1");

	if (array->shape()[0] == 0) {
		return false;
	}

	ref[0] = 0;
	return true;
}

Variant NDArray::_iter_next(const Array& p_iter) {
	ERR_FAIL_COND_V_MSG(array->shape().empty(), false, "iteration over a 0-d array");

	Array ref = p_iter;
	ERR_FAIL_COND_V_MSG(ref.size() != 1, false, "size of iterator cache must be 1");

	const auto size = array->shape()[0];
	int pos = ref[0];
	ERR_FAIL_COND_V_MSG(pos < 0 || pos >= size, false, "iterator out of bounds");

	pos += 1;
	ref[0] = pos;

	return pos != size;
}

Variant NDArray::_iter_get(const Variant& p_iter) {
	ERR_FAIL_COND_V_MSG(array->shape().empty(), false, "iteration over a 0-d array");

	int pos = p_iter;
	const auto size = array->shape()[0];
	if (pos < 0 || pos >= size) { return {}; }
	ERR_FAIL_COND_V_MSG(pos < 0 || pos >= size, false, "iterator out of bounds");

	// We checked for the shape size, the next should not fail.
	const auto result = array->sliced({ pos });
	return { memnew(NDArray(result)) };
}

Ref<NDArray> NDArray::as_type(const va::DType dtype) const {
	const auto result = ndarray_as_dtype(*this, dtype);
	return { memnew(NDArray(result)) };
}

Ref<NDArray> NDArray::copy() const {
	const auto result = va::copy(va::store::default_allocator, array->data);
	return { memnew(NDArray(result)) };
}

Ref<NDArray> NDArray::transpose(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	try {
		switch (arg_count) {
			case 0:
				return { memnew(NDArray(va::transpose(*array))) };
			case 1: {
				const auto axes = variant_to_axes(*args[0]);
				return { memnew(NDArray(va::transpose(*array, axes))) };
			}
			default: {
				const auto axes = variants_to_axes(args, arg_count, error);
				return { memnew(NDArray(va::transpose(*array, axes))) };
			}
		}
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> NDArray::flatten() const {
	try {
		const auto result = va::flatten(va::store::default_allocator, *array);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

va::VData get_write(va::VArray& array, const xt::xstrided_slice_vector& sv) {
	// TODO Need to prepare_write()?
	return array.sliced_data(sv);
}

va::VData get_write(va::VArray& array, const single_axis_slice& sv) {
	// TODO Need to prepare_write()?
	return array.sliced_data(std::get<0>(sv), std::get<1>(sv));
}

static void set_vdata_variant(va::VData& data, const Variant& value) {
	switch (value.get_type()) {
		case Variant::BOOL:
			va::assign(data, static_cast<bool>(value));
			return;
		case Variant::INT:
			va::assign(data, static_cast<int64_t>(value));
			return;
		case Variant::FLOAT:
			va::assign(data, static_cast<double_t>(value));
			return;
		// TODO We could optimize more assignments of literals.
		//  Just need to figure out how, ideally without duplicating code - as_array already does much type checking work.
		default:
			const auto value_ = variant_as_array(value);
			va::assign(data, value_->data);
			return;
	}
}

void NDArray::set(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	ERR_FAIL_COND_MSG(arg_count < 1, "At least one argument must be passed to NDarray->set(). Ignoring assignment.");

	try {
		const Variant& value = *args[0];

		std::visit([this, &value](auto slice) {
			using T = std::decay_t<decltype(slice)>;

			if constexpr (std::is_same_v<T, xt::xstrided_slice_vector>) {
				// See if we can't make it a single-value assign.
				// This should be faster than fill-assign, due to various overheads.
				// TODO We should figure out if this is fast enough. If not, we'll need a set_scalar function.
				if (slice.size() == array->dimension()) {
					auto as_axes = slice_vector_to_axes_list(slice);
					if (as_axes != std::nullopt) {
						array->prepare_write();
						va::set_single_value(array->data, *as_axes, variant_to_vscalar(value));
						return;
					}
				}
			}

			if constexpr (std::is_same_v<T, std::nullptr_t>) {
				array->prepare_write();
				set_vdata_variant(array->data, value);
			}
			else if constexpr (std::is_same_v<T, xt::xstrided_slice_vector> || std::is_same_v<T, single_axis_slice>) {
				auto data = get_write(*array, slice);
				set_vdata_variant(data, value);
			}
			else if constexpr (std::is_same_v<T, SliceIndexList>) {
				array->prepare_write();

				const auto value_ = variant_as_array(value);
				va::set_at_indices(array->data, slice.index_list->data, value_->data);
			}
			else {
				// Mask
				array->prepare_write();
				auto compute = array->data;

				switch (value.get_type()) {
					case Variant::BOOL:
						va::set_at_mask(compute, slice.mask->data, static_cast<bool>(value));
						return;
					case Variant::INT:
						va::set_at_mask(compute, slice.mask->data, static_cast<int64_t>(value));
						return;
					case Variant::FLOAT:
						va::set_at_mask(compute, slice.mask->data, static_cast<double_t>(value));
						return;
					default:
						const auto value_ = variant_as_array(value);
						va::set_at_mask(compute, slice.mask->data, value_->data);
						return;
				}
			}
		}, variants_to_slice_variant(args + 1, arg_count - 1, error));
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_MSG(error.what());
	}
}

std::shared_ptr<va::VArray> get_varray(const std::shared_ptr<va::VArray>& array, const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return std::visit([&array](auto slice) -> std::shared_ptr<va::VArray> {
		using T = std::decay_t<decltype(slice)>;

		if constexpr (std::is_same_v<T, xt::xstrided_slice_vector>) {
			return array->sliced(slice);
		}
		else if constexpr (std::is_same_v<T, std::nullptr_t>) {
			return array;
		}
		else if constexpr (std::is_same_v<T, single_axis_slice>) {
			return array->sliced(std::get<0>(slice), std::get<1>(slice));
		}
		else if constexpr (std::is_same_v<T, SliceIndexList>) {
			return va::get_at_indices(va::store::default_allocator, array->data, slice.index_list->data);
		}
		else {
			// Mask
			return va::get_at_mask(va::store::default_allocator, array->data, slice.mask->data);
		}
	}, variants_to_slice_variant(args, arg_count, error));
}

Ref<NDArray> NDArray::get(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	try {
		return { memnew(NDArray(get_varray(array, args, arg_count, error))) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(Ref<NDArray>(), error.what());
	}
}

bool NDArray::get_bool(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	try {
		auto axes = variants_to_axes(args, arg_count, error);
		return va::static_cast_scalar<bool>(va::get_single_value(array->data, axes));
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

int64_t NDArray::get_int(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	try {
		auto axes = variants_to_axes(args, arg_count, error);
		return va::static_cast_scalar<int64_t>(va::get_single_value(array->data, axes));
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

double_t NDArray::get_float(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	try {
		auto axes = variants_to_axes(args, arg_count, error);
		return va::static_cast_scalar<double_t>(va::get_single_value(array->data, axes));
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

template <typename T>
T get_slice_tensor(const std::shared_ptr<va::VArray>& array, const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	try {
		const auto slice = get_varray(array, args, arg_count, error);
		return numdot::to_variant_tensor<T>(slice->data);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Vector2 NDArray::get_vector2(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Vector2>(array, args, arg_count, error);
}

Vector3 NDArray::get_vector3(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Vector3>(array, args, arg_count, error);
}

Vector4 NDArray::get_vector4(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Vector4>(array, args, arg_count, error);
}

Vector2i NDArray::get_vector2i(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Vector2i>(array, args, arg_count, error);
}

Vector3i NDArray::get_vector3i(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Vector3i>(array, args, arg_count, error);
}

Vector4i NDArray::get_vector4i(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Vector4i>(array, args, arg_count, error);
}

Color NDArray::get_color(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Color>(array, args, arg_count, error);
}

Quaternion NDArray::get_quaternion(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Quaternion>(array, args, arg_count, error);
}

Plane NDArray::get_plane(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Plane>(array, args, arg_count, error);
}

Basis NDArray::get_basis(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Basis>(array, args, arg_count, error);
}

Projection NDArray::get_projection(const Variant** args, GDExtensionInt arg_count, GDExtensionCallError& error) {
	return get_slice_tensor<Projection>(array, args, arg_count, error);
}

bool NDArray::to_bool() const { return static_cast<bool>(*this); }
int64_t NDArray::to_int() const { return static_cast<int64_t>(*this); }
double_t NDArray::to_float() const { return static_cast<double_t>(*this); }

Vector2 NDArray::to_vector2() const {
	return numdot::to_variant_tensor<Vector2>(array->data);
}

Vector3 NDArray::to_vector3() const {
	return numdot::to_variant_tensor<Vector3>(array->data);
}

Vector4 NDArray::to_vector4() const {
	return numdot::to_variant_tensor<Vector4>(array->data);
}

Vector2i NDArray::to_vector2i() const {
	return numdot::to_variant_tensor<Vector2i>(array->data);
}

Vector3i NDArray::to_vector3i() const {
	return numdot::to_variant_tensor<Vector3i>(array->data);
}

Vector4i NDArray::to_vector4i() const {
	return numdot::to_variant_tensor<Vector4i>(array->data);
}

Color NDArray::to_color() const {
	return numdot::to_variant_tensor<Color>(array->data);
}

Quaternion NDArray::to_quaternion() const {
	return numdot::to_variant_tensor<Quaternion>(array->data);
}

Plane NDArray::to_plane() const {
	return numdot::to_variant_tensor<Plane>(array->data);
}

Basis NDArray::to_basis() const {
	return numdot::to_variant_tensor<Basis>(array->data);
}

Projection NDArray::to_projection() const {
	return numdot::to_variant_tensor<Projection>(array->data);
}

PackedFloat32Array NDArray::to_packed_float32_array() const {
	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedFloat32Array*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedFloat32Array, float_t>(array->data);
}

PackedFloat64Array NDArray::to_packed_float64_array() const {
	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedFloat64Array*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedFloat64Array, double_t>(array->data);
}

PackedByteArray NDArray::to_packed_byte_array() const {
	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedByteArray*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedByteArray, uint8_t>(array->data);
}

PackedInt32Array NDArray::to_packed_int32_array() const {
	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedInt32Array*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedInt32Array, int32_t>(array->data);
}

PackedInt64Array NDArray::to_packed_int64_array() const {
	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedInt64Array*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedInt64Array, int64_t>(array->data);
}

PackedVector2Array NDArray::to_packed_vector2_array() const {
	ERR_FAIL_COND_V_MSG(array->dimension() != 2, {}, "flatten the array before converting to packed");
	ERR_FAIL_COND_V_MSG(array->shape()[1] != 2, {}, "final array dimension must be size 2");

	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedVector2Array*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedVector2Array, real_t, 2>(array->data);
}

PackedVector3Array NDArray::to_packed_vector3_array() const {
	ERR_FAIL_COND_V_MSG(array->dimension() != 2, {}, "flatten the array before converting to packed");
	ERR_FAIL_COND_V_MSG(array->shape()[1] != 3, {}, "final array dimension must be size 2");

	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedVector3Array*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedVector3Array, real_t, 3>(array->data);
}

PackedVector4Array NDArray::to_packed_vector4_array() const {
	ERR_FAIL_COND_V_MSG(array->dimension() != 2, {}, "flatten the array before converting to packed");
	ERR_FAIL_COND_V_MSG(array->shape()[1] != 4, {}, "final array dimension must be size 2");

	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedVector4Array*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedVector4Array, real_t, 4>(array->data);
}

PackedColorArray NDArray::to_packed_color_array() const {
	ERR_FAIL_COND_V_MSG(array->dimension() != 2, {}, "flatten the array before converting to packed");
	ERR_FAIL_COND_V_MSG(array->shape()[1] != 4, {}, "final array dimension must be size 2");

	if (array->is_contiguous() && array->is_full_view()) {
		if (auto* store = dynamic_cast<numdot::VStorePackedColorArray*>(&*array->store)) {
			return store->array;
		}
	}

	return numdot::to_packed<PackedColorArray, float_t, 4>(array->data);
}

TypedArray<NDArray> NDArray::to_godot_array() const {
	ERR_FAIL_COND_V_MSG(array->dimension() == 0, {}, "can't slice a 0-dimension vector");

	auto godot_array = TypedArray<NDArray>();
	const std::size_t outer_dim_size = array->shape()[0];
	godot_array.resize(static_cast<int64_t>(outer_dim_size));
	for (std::size_t i = 0; i < outer_dim_size; i++) {
		xt::xstrided_slice_vector idx = {i};
		godot_array[static_cast<int64_t>(i)] = { memnew(NDArray(array->sliced(idx))) };
	}
	return godot_array;
}

template<typename Visitor, typename... Args>
void map_variants_as_arrays_inplace(Visitor&& visitor, va::VArray& target, const Args&... args) {
	try {
		target.prepare_write();
		std::forward<Visitor>(visitor)(&target.data, variant_as_array(args)...);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_MSG(error.what());
	}
}

template<typename Visitor, typename VisitorNoaxes, typename... Args>
inline void reduction_inplace(Visitor&& visitor, VisitorNoaxes&& visitor_noaxes, va::VArray& target, const Variant& axes, const Args&... args) {
	try {
		target.prepare_write();

		if (axes.get_type() == Variant::NIL) {
			const va::VScalar result = std::forward<VisitorNoaxes>(visitor_noaxes)(variant_as_array(args)...);
			va::assign(target.data, result);
			return;
		}

		const auto axes_ = variant_to_axes(axes);

		std::forward<Visitor>(visitor)(static_cast<va::VArrayTarget>(&target.data), axes_, variant_as_array(args)...);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_MSG(error.what());
	}
}

#define VARRAY_MAP1(func, varray1) \
	map_variants_as_arrays_inplace([this](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& varray) {\
        va::func(va::store::default_allocator, target, varray->data);\
    }, *this->array, (varray1));\
    return {this}

#define VARRAY_MAP2(func, varray1, varray2) \
	map_variants_as_arrays_inplace([this](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b) {\
        va::func(va::store::default_allocator, target, a->data, b->data);\
    }, *this->array, (varray1), (varray2));\
    return {this}

#define VARRAY_MAP3(func, varray1, varray2, varray3) \
	map_variants_as_arrays_inplace([this](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b, const std::shared_ptr<va::VArray>& c) {\
        va::func(va::store::default_allocator, target, a->data, b->data, c->data);\
    }, *this->array, (varray1), (varray2), (varray3));\
    return {this}

#define REDUCTION1(func, varray1, axes1) \
	reduction_inplace([this](const va::VArrayTarget& target, const va::axes_type& axes, const std::shared_ptr<va::VArray>& array) {\
		va::func(va::store::default_allocator, target, array->data, axes);\
	}, [this](const std::shared_ptr<va::VArray>& array) {\
		return va::func(array->data);\
	}, *this->array, (axes1), (varray1));\
	return {this}

#define REDUCTION2(func, varray1, varray2, axes1) \
	reduction_inplace([this](const va::VArrayTarget& target, const va::axes_type& axes, const std::shared_ptr<va::VArray>& carray1, const std::shared_ptr<va::VArray>& carray2) {\
		va::func(va::store::default_allocator, target, carray1->data, carray2->data, axes);\
	}, [this](const std::shared_ptr<va::VArray>& carray1, const std::shared_ptr<va::VArray>& carray2) {\
		return va::func(carray1->data, carray2->data);\
	}, *this->array, (axes1), (varray1), (varray2));\
	return {this}

Ref<NDArray> NDArray::assign_conjugate(const Variant& a) {
	VARRAY_MAP1(conjugate, a);
}

Ref<NDArray> NDArray::assign_angle(const Variant& a) {
	map_variants_as_arrays_inplace(
		[this](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& varray) {
			va::angle(va::store::default_allocator, target, varray);
		}, *this->array, a);
	return {this};
}

Ref<NDArray> NDArray::assign_positive(const Variant& a) {
	VARRAY_MAP1(positive, a);
}

Ref<NDArray> NDArray::assign_negative(const Variant& a) {
	VARRAY_MAP1(negative, a);
}

Ref<NDArray> NDArray::assign_add(const Variant& a, const Variant& b) {
	// godot::UtilityFunctions::print(value);
	VARRAY_MAP2(add, a, b);
}

Ref<NDArray> NDArray::assign_subtract(const Variant& a, const Variant& b) {
	VARRAY_MAP2(subtract, a, b);
}

Ref<NDArray> NDArray::assign_multiply(const Variant& a, const Variant& b) {
	VARRAY_MAP2(multiply, a, b);
}

Ref<NDArray> NDArray::assign_divide(const Variant& a, const Variant& b) {
	VARRAY_MAP2(divide, a, b);
}

Ref<NDArray> NDArray::assign_remainder(const Variant& a, const Variant& b) {
	VARRAY_MAP2(remainder, a, b);
}

Ref<NDArray> NDArray::assign_pow(const Variant& a, const Variant& b) {
	VARRAY_MAP2(pow, a, b);
}

Ref<NDArray> NDArray::assign_minimum(const Variant& a, const Variant& b) {
	VARRAY_MAP2(minimum, a, b);
}

Ref<NDArray> NDArray::assign_maximum(const Variant& a, const Variant& b) {
	VARRAY_MAP2(maximum, a, b);
}

Ref<NDArray> NDArray::assign_clip(const Variant& a, const Variant& min, const Variant& max) {
	VARRAY_MAP3(clip, a, min, max);
}

Ref<NDArray> NDArray::assign_sign(const Variant& a) {
	VARRAY_MAP1(sign, a);
}

Ref<NDArray> NDArray::assign_abs(const Variant& a) {
	VARRAY_MAP1(abs, a);
}

Ref<NDArray> NDArray::assign_square(const Variant& a) {
	VARRAY_MAP1(square, a);
}

Ref<NDArray> NDArray::assign_sqrt(const Variant& a) {
	VARRAY_MAP1(sqrt, a);
}

Ref<NDArray> NDArray::assign_exp(const Variant& a) {
	VARRAY_MAP1(exp, a);
}

Ref<NDArray> NDArray::assign_log(const Variant& a) {
	VARRAY_MAP1(log, a);
}

Ref<NDArray> NDArray::assign_rad2deg(const Variant& a) {
	VARRAY_MAP1(rad2deg, a);
}

Ref<NDArray> NDArray::assign_deg2rad(const Variant& a) {
	VARRAY_MAP1(deg2rad, a);
}

Ref<NDArray> NDArray::assign_sin(const Variant& a) {
	VARRAY_MAP1(sin, a);
}

Ref<NDArray> NDArray::assign_cos(const Variant& a) {
	VARRAY_MAP1(cos, a);
}

Ref<NDArray> NDArray::assign_tan(const Variant& a) {
	VARRAY_MAP1(tan, a);
}

Ref<NDArray> NDArray::assign_asin(const Variant& a) {
	VARRAY_MAP1(asin, a);
}

Ref<NDArray> NDArray::assign_acos(const Variant& a) {
	VARRAY_MAP1(acos, a);
}

Ref<NDArray> NDArray::assign_atan(const Variant& a) {
	VARRAY_MAP1(atan, a);
}

Ref<NDArray> NDArray::assign_atan2(const Variant& x1, const Variant& x2) {
	VARRAY_MAP2(atan2, x1, x2);
}

Ref<NDArray> NDArray::assign_sinh(const Variant& a) {
	VARRAY_MAP1(sinh, a);
}

Ref<NDArray> NDArray::assign_cosh(const Variant& a) {
	VARRAY_MAP1(cosh, a);
}

Ref<NDArray> NDArray::assign_tanh(const Variant& a) {
	VARRAY_MAP1(tanh, a);
}

Ref<NDArray> NDArray::assign_asinh(const Variant& a) {
	VARRAY_MAP1(asinh, a);
}

Ref<NDArray> NDArray::assign_acosh(const Variant& a) {
	VARRAY_MAP1(acosh, a);
}

Ref<NDArray> NDArray::assign_atanh(const Variant& a) {
	VARRAY_MAP1(atanh, a);
}

Ref<NDArray> NDArray::assign_sum(const Variant& a, const Variant& axes) {
	REDUCTION1(sum, a, axes);
}

Ref<NDArray> NDArray::assign_prod(const Variant& a, const Variant& axes) {
	REDUCTION1(prod, a, axes);
}

Ref<NDArray> NDArray::assign_mean(const Variant& a, const Variant& axes) {
	REDUCTION1(mean, a, axes);
}

Ref<NDArray> NDArray::assign_variance(const Variant& a, const Variant& axes) {
	REDUCTION1(variance, a, axes);
}

Ref<NDArray> NDArray::assign_standard_deviation(const Variant& a, const Variant& axes) {
	REDUCTION1(standard_deviation, a, axes);
}

Ref<NDArray> NDArray::assign_max(const Variant& a, const Variant& axes) {
	REDUCTION1(max, a, axes);
}

Ref<NDArray> NDArray::assign_min(const Variant& a, const Variant& axes) {
	REDUCTION1(min, a, axes);
}

Ref<NDArray> NDArray::assign_norm(const Variant& a, const Variant& ord, const Variant& axes) {
	switch (ord.get_type()) {
		case Variant::INT:
			switch (static_cast<int64_t>(ord)) {
				case 0:
					REDUCTION1(norm_l0, a, axes);
				case 1:
					REDUCTION1(norm_l1, a, axes);
				case 2:
					REDUCTION1(norm_l2, a, axes);
				default:
					break;
			}
		case Variant::FLOAT:
			if (std::isinf(static_cast<double_t>(ord))) {
				REDUCTION1(norm_linf, a, axes);
			}
		default:
			break;
	}

	ERR_FAIL_V_MSG({this}, "This norm is currently not supported");
}

Ref<NDArray> NDArray::assign_floor(const Variant& a) {
	VARRAY_MAP1(floor, a);
}

Ref<NDArray> NDArray::assign_ceil(const Variant& a) {
	VARRAY_MAP1(ceil, a);
}

Ref<NDArray> NDArray::assign_round(const Variant& a) {
	VARRAY_MAP1(round, a);
}

Ref<NDArray> NDArray::assign_trunc(const Variant& a) {
	VARRAY_MAP1(trunc, a);
}

Ref<NDArray> NDArray::assign_rint(const Variant& a) {
	// Actually uses nearbyint because rint can throw, which is undesirable in our case, and unlike numpy's behavior.
	VARRAY_MAP1(nearbyint, a);
}

Ref<NDArray> NDArray::assign_equal(const Variant& a, const Variant& b) {
	VARRAY_MAP2(equal_to, a, b);
}

Ref<NDArray> NDArray::assign_not_equal(const Variant& a, const Variant& b) {
	VARRAY_MAP2(not_equal_to, a, b);
}

Ref<NDArray> NDArray::assign_greater(const Variant& a, const Variant& b) {
	VARRAY_MAP2(greater, a, b);
}

Ref<NDArray> NDArray::assign_greater_equal(const Variant& a, const Variant& b) {
	VARRAY_MAP2(greater_equal, a, b);
}

Ref<NDArray> NDArray::assign_less(const Variant& a, const Variant& b) {
	VARRAY_MAP2(less, a, b);
}

Ref<NDArray> NDArray::assign_less_equal(const Variant& a, const Variant& b) {
	VARRAY_MAP2(less_equal, a, b);
}

Ref<NDArray> NDArray::assign_logical_and(const Variant& a, const Variant& b) {
	VARRAY_MAP2(logical_and, a, b);
}

Ref<NDArray> NDArray::assign_logical_or(const Variant& a, const Variant& b) {
	VARRAY_MAP2(logical_or, a, b);
}

Ref<NDArray> NDArray::assign_logical_xor(const Variant& a, const Variant& b) {
	VARRAY_MAP2(logical_xor, a, b);
}

Ref<NDArray> NDArray::assign_logical_not(const Variant& a) {
	VARRAY_MAP1(logical_not, a);
}

Ref<NDArray> NDArray::assign_all(const Variant& a, const Variant& axes) {
	REDUCTION1(all, a, axes);
}

Ref<NDArray> NDArray::assign_any(const Variant& a, const Variant& axes) {
	REDUCTION1(any, a, axes);
}

Ref<NDArray> NDArray::assign_bitwise_and(const Variant& a, const Variant& b) {
	VARRAY_MAP2(bitwise_and, a, b);
}

Ref<NDArray> NDArray::assign_bitwise_or(const Variant& a, const Variant& b) {
	VARRAY_MAP2(bitwise_or, a, b);
}

Ref<NDArray> NDArray::assign_bitwise_xor(const Variant& a, const Variant& b) {
	VARRAY_MAP2(bitwise_xor, a, b);
}

Ref<NDArray> NDArray::assign_bitwise_not(const Variant& a) {
	VARRAY_MAP1(bitwise_not, a);
}

Ref<NDArray> NDArray::assign_bitwise_left_shift(const Variant& a, const Variant& b) {
	VARRAY_MAP2(bitwise_left_shift, a, b);
}

Ref<NDArray> NDArray::assign_bitwise_right_shift(const Variant& a, const Variant& b) {
	VARRAY_MAP2(bitwise_right_shift, a, b);
}

Ref<NDArray> NDArray::assign_dot(const Variant& a, const Variant& b) {
	VARRAY_MAP2(dot, a, b);
}

Ref<NDArray> NDArray::assign_reduce_dot(const Variant& a, const Variant& b, const Variant& axes) {
	REDUCTION2(reduce_dot, a, b, axes);
}

Ref<NDArray> NDArray::assign_matmul(const Variant& a, const Variant& b) {
	VARRAY_MAP2(matmul, a, b);
}

Ref<NDArray> NDArray::assign_cross(const Variant& a, const Variant& b, int64_t axisa, int64_t axisb, int64_t axisc) {
	map_variants_as_arrays_inplace([this, axisa, axisb, axisc](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b) {
		va::cross(va::store::default_allocator, target, a->data, b->data, axisa, axisb, axisc);
	}, *this->array, a, b);\
	return {this};
}

Ref<NDArray> NDArray::assign_convolve(const Variant& array, const Variant& kernel) {
	map_variants_as_arrays_inplace([this](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b) {\
		va::convolve(va::store::default_allocator, target, *a, *b);\
	}, *this->array, array, kernel);\
	return {this};
}

#define CONVERT_TO_SCALAR(type)\
try {\
	return static_cast<type>(*array);\
}\
catch (std::runtime_error& error) {\
	ERR_FAIL_V_MSG(false, error.what());\
}

NDArray::operator bool() const { CONVERT_TO_SCALAR(bool); }
NDArray::operator int64_t() const { CONVERT_TO_SCALAR(int64_t); }
NDArray::operator int32_t() const { CONVERT_TO_SCALAR(int32_t); }
NDArray::operator int16_t() const { CONVERT_TO_SCALAR(int16_t); }
NDArray::operator int8_t() const { CONVERT_TO_SCALAR(int8_t); }
NDArray::operator uint64_t() const { CONVERT_TO_SCALAR(uint64_t); }
NDArray::operator uint32_t() const { CONVERT_TO_SCALAR(uint32_t); }
NDArray::operator uint16_t() const { CONVERT_TO_SCALAR(uint16_t); }
NDArray::operator uint8_t() const { CONVERT_TO_SCALAR(uint8_t); }
NDArray::operator double() const { CONVERT_TO_SCALAR(double); }
NDArray::operator float() const { CONVERT_TO_SCALAR(float); }

#undef VARRAY_MAP1
#undef VARRAY_MAP2
#undef VARRAY_MAP3
#undef REDUCTION1
#undef REDUCTION2
#undef TRY_CONVERT
#undef CONVERT_TO_SCALAR
