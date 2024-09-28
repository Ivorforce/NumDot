#include "ndarray.h"

#include <gdconvert/conversion_ints.h>             // for variants_to_axes
#include <vatensor/comparison.h>                   // for equal_to, greater
#include <vatensor/linalg.h>                       // for reduce_dot, dot
#include <vatensor/logical.h>                      // for logical_and, logic...
#include <vatensor/reduce.h>                       // for all, any, max, mean
#include <vatensor/trigonometry.h>                 // for acos, acosh, asin
#include <vatensor/vassign.h>                      // for assign
#include <vatensor/vmath.h>                        // for abs, add, clip
#include <algorithm>                               // for copy
#include <cstddef>                                 // for size_t
#include <stdexcept>                               // for runtime_error
#include <variant>                                 // for visit
#include "gdconvert/conversion_array.h"            // for fill_c_array_flat
#include "gdconvert/conversion_slice.h"            // for variants_to_slice_...
#include "gdconvert/conversion_string.h"           // for xt_to_string
#include "godot_cpp/classes/global_constants.hpp"  // for MethodFlags
#include "godot_cpp/core/class_db.hpp"             // for D_METHOD, ClassDB
#include "godot_cpp/core/error_macros.hpp"         // for ERR_FAIL_COND_V_MSG
#include "godot_cpp/core/memory.hpp"               // for _post_initialize
#include "godot_cpp/variant/string_name.hpp"       // for StringName
#include "godot_cpp/variant/variant.hpp"           // for Variant
#include "nd.h"                                    // for nd
#include "vatensor/round.h"                        // for ceil, floor, nearb...
#include "vatensor/varray.h"                       // for VArray, VArrayTarget
#include "xtensor/xiterator.hpp"                   // for operator==
#include "xtensor/xstrided_view.hpp"               // for xstrided_slice_vector
#include "xtl/xiterator_base.hpp"                  // for operator!=

using namespace godot;

void NDArray::_bind_methods() {
	BIND_ENUM_CONSTANT(RowMajor);
	BIND_ENUM_CONSTANT(ColumnMajor);
	BIND_ENUM_CONSTANT(Dynamic);

	godot::ClassDB::bind_method(D_METHOD("dtype"), &NDArray::dtype);
	godot::ClassDB::bind_method(D_METHOD("shape"), &NDArray::shape);
	godot::ClassDB::bind_method(D_METHOD("size"), &NDArray::size);
	godot::ClassDB::bind_method(D_METHOD("array_size_in_bytes"), &NDArray::array_size_in_bytes);
	godot::ClassDB::bind_method(D_METHOD("ndim"), &NDArray::ndim);
	godot::ClassDB::bind_method(D_METHOD("strides"), &NDArray::strides);
	godot::ClassDB::bind_method(D_METHOD("strides_layout"), &NDArray::strides_layout);
	godot::ClassDB::bind_method(D_METHOD("strides_offset"), &NDArray::strides_offset);

	ClassDB::bind_method(D_METHOD("_iter_init"), &NDArray::_iter_init);
	ClassDB::bind_method(D_METHOD("_iter_get"), &NDArray::_iter_get);
	ClassDB::bind_method(D_METHOD("_iter_next"), &NDArray::_iter_next);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "set", &NDArray::set);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "get", &NDArray::get);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "get_bool", &NDArray::get_bool);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "get_int", &NDArray::get_int);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "get_float", &NDArray::get_float);

	godot::ClassDB::bind_method(D_METHOD("as_type", "type"), &NDArray::as_type);

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

	godot::ClassDB::bind_method(D_METHOD("assign_sum", "a", "axes"), &NDArray::assign_sum, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_prod", "a", "axes"), &NDArray::assign_prod, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_mean", "a", "axes"), &NDArray::assign_mean, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_var", "a", "axes"), &NDArray::assign_var, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_std", "a", "axes"), &NDArray::assign_std, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_max", "a", "axes"), &NDArray::assign_max, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_min", "a", "axes"), &NDArray::assign_min, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_norm", "a", "ord", "axes"), &NDArray::assign_norm, DEFVAL(nullptr), DEFVAL(2), DEFVAL(nullptr));

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
    godot::ClassDB::bind_method(D_METHOD("assign_all", "a", "axes"), &NDArray::assign_all, DEFVAL(nullptr), DEFVAL(nullptr));
    godot::ClassDB::bind_method(D_METHOD("assign_any", "a", "axes"), &NDArray::assign_any, DEFVAL(nullptr), DEFVAL(nullptr));

	godot::ClassDB::bind_method(D_METHOD("assign_dot", "a", "b"), &NDArray::assign_dot);
	godot::ClassDB::bind_method(D_METHOD("assign_reduce_dot", "a", "b", "axes"), &NDArray::assign_reduce_dot, DEFVAL(nullptr), DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_method(D_METHOD("assign_matmul", "a", "b"), &NDArray::assign_matmul);
}

NDArray::NDArray() = default;

NDArray::~NDArray() = default;

String NDArray::_to_string() const {
	return std::visit([](auto&& arg){ return xt_to_string(arg); }, array.compute_read());
}

va::DType NDArray::dtype() const {
	return array.dtype();
}

PackedInt64Array NDArray::shape() const {
	PackedInt64Array shape;
	shape.resize(static_cast<int64_t>(array.shape.size()));
	std::copy(array.shape.begin(), array.shape.end(), shape.ptrw());
	return shape;
}

uint64_t NDArray::size() const {
	return array.size();
}

uint64_t NDArray::array_size_in_bytes() const {
	return array.size_of_array_in_bytes();
}

uint64_t NDArray::ndim() const {
	return array.dimension();
}

PackedInt64Array NDArray::strides() const {
	PackedInt64Array strides;
	strides.resize(static_cast<int64_t>(array.strides.size()));
	std::copy(array.strides.begin(), array.strides.end(), strides.ptrw());
	return strides;
}

NDArray::Layout NDArray::strides_layout() const {
	switch (array.layout) {
		case xt::layout_type::row_major:
			return NDArray::Layout::RowMajor;
		case xt::layout_type::column_major:
			return NDArray::Layout::ColumnMajor;
		case xt::layout_type::dynamic:
			return NDArray::Layout::Dynamic;
		case xt::layout_type::any:
			ERR_FAIL_V_MSG(NDArray::Dynamic, "Unrecognized 'any' layout found.");
		default: ;
			ERR_FAIL_V_MSG(NDArray::Dynamic, "Unrecognized layout found.");
	}
}

uint64_t NDArray::strides_offset() const {
	return static_cast<uint64_t>(array.offset);
}

Variant NDArray::_iter_init(const Array &p_iter) {
	ERR_FAIL_COND_V_MSG(array.shape.empty(), false, "iteration over a 0-d array");

	Array ref = p_iter;
	ERR_FAIL_COND_V_MSG(ref.size() != 1, false, "size of iterator cache must be 1");

	if (array.shape[0] == 0) {
		return false;
	}

	ref[0] = 0;
	return true;
}

Variant NDArray::_iter_next(const Array &p_iter) {
	ERR_FAIL_COND_V_MSG(array.shape.empty(), false, "iteration over a 0-d array");

	Array ref = p_iter;
	ERR_FAIL_COND_V_MSG(ref.size() != 1, false, "size of iterator cache must be 1");

	const auto size = array.shape[0];
	int pos = ref[0];
	ERR_FAIL_COND_V_MSG(pos < 0 || pos >= size, false, "iterator out of bounds");

	pos += 1;
	ref[0] = pos;

	return pos != size;
}

Variant NDArray::_iter_get(const Variant &p_iter) {
	ERR_FAIL_COND_V_MSG(array.shape.empty(), false, "iteration over a 0-d array");

	int pos = p_iter;
	const auto size = array.shape[0];
	if (pos < 0 || pos >= size) { return {}; }
	ERR_FAIL_COND_V_MSG(pos < 0 || pos >= size, false, "iterator out of bounds");

	// We checked for the shape size, the next should not fail.
	const auto result = array.slice({pos});
	return {memnew(NDArray(result))};
}

Variant NDArray::as_type(va::DType dtype) const {
	return nd::as_array(this, dtype);
}

void NDArray::set(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	ERR_FAIL_COND_MSG(arg_count < 1, "At least one argument must be passed to NDArray.set(). Ignoring assignment.");

	try {
		const Variant &value = *args[0];
		// todo don't need slices if arg_count == 1
		auto compute = arg_count == 1
			? array.compute_write()
			: array.compute_write(variants_to_slice_vector(args + 1, arg_count - 1, error));

		switch (value.get_type()) {
			case Variant::BOOL:
				va::assign(compute, static_cast<bool>(value));
				return;
			case Variant::INT:
				va::assign(compute, static_cast<int64_t>(value));
				return;
			case Variant::FLOAT:
				va::assign(compute, static_cast<double_t>(value));
				return;
			// TODO We could optimize more assignments of literals.
			//  Just need to figure out how, ideally without duplicating code - as_array already does much type checking work.
			default:
				const auto a_ = variant_as_array(value).compute_read();

				va::assign(compute, a_);
				return;
		}
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_MSG(error.what());
	}
}

Ref<NDArray> NDArray::get(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	try {
		xt::xstrided_slice_vector sv = variants_to_slice_vector(args, arg_count, error);

		const auto result = array.slice(sv);
		return {memnew(NDArray(result))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(Ref<NDArray>(), error.what());
	}
}

bool NDArray::get_bool(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	try {
		return va::scalar_to_type<bool>(array.get_scalar(variants_to_axes(args, arg_count, error)));
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

int64_t NDArray::get_int(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	try {
		return va::scalar_to_type<int64_t>(array.get_scalar(variants_to_axes(args, arg_count, error)));
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

double_t NDArray::get_float(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	try {
		return va::scalar_to_type<double_t>(array.get_scalar(variants_to_axes(args, arg_count, error)));
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

bool NDArray::to_bool() const { return static_cast<bool>(*this);}
int64_t NDArray::to_int() const { return static_cast<int64_t>(*this); }
double_t NDArray::to_float() const { return static_cast<double_t>(*this); }

Vector2 NDArray::to_vector2() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
    ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to vector");
    ERR_FAIL_COND_V_MSG(array.shape[0] != 2, {}, "array dimension must be size 2");

    Vector2 vector;
    fill_c_array_flat(&vector.coord[0], array.compute_read());

    return vector;
#endif
}

Vector3 NDArray::to_vector3() const{
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
    ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to vector");
    ERR_FAIL_COND_V_MSG(array.shape[0] != 3, {}, "array dimension must be size 3");

    Vector3 vector;
    fill_c_array_flat(&vector.coord[0], array.compute_read());

    return vector;
#endif
}

Vector4 NDArray::to_vector4() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
    ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to vector");
    ERR_FAIL_COND_V_MSG(array.shape[0] != 4, {}, "array dimension must be size 4");

    Vector4 vector;
    fill_c_array_flat(&vector.components[0], array.compute_read());

    return vector;
#endif
}

Vector2i NDArray::to_vector2i() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
    ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to vector");
    ERR_FAIL_COND_V_MSG(array.shape[0] != 2, {}, "array dimension must be size 2");

    Vector2i vector;
    fill_c_array_flat(&vector.coord[0], array.compute_read());

    return vector;
#endif
}

Vector3i NDArray::to_vector3i() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
    ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to vector");
    ERR_FAIL_COND_V_MSG(array.shape[0] != 3, {}, "array dimension must be size 3");

    Vector3i vector;
    fill_c_array_flat(&vector.coord[0], array.compute_read());

    return vector;
#endif
}

Vector4i NDArray::to_vector4i() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
    ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to vector");
    ERR_FAIL_COND_V_MSG(array.shape[0] != 4, {}, "array dimension must be size 4");

    Vector4i vector;
    fill_c_array_flat(&vector.coord[0], array.compute_read());

    return vector;
#endif
}

Color NDArray::to_color() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
    ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to color");
    ERR_FAIL_COND_V_MSG(array.shape[0] != 4, {}, "array dimension must be size 4");

    Color vector;
    fill_c_array_flat(&vector.components[0], array.compute_read());

    return vector;
#endif
}

PackedFloat32Array NDArray::to_packed_float32_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to packed");

	PackedFloat32Array packed;
	packed.resize(static_cast<int64_t>(array.shape[0]));
	fill_c_array_flat(packed.ptrw(), array.compute_read());

	return packed;
#endif
}

PackedFloat64Array NDArray::to_packed_float64_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to packed");

	PackedFloat64Array packed;
	packed.resize(static_cast<int64_t>(array.shape[0]));
	fill_c_array_flat(packed.ptrw(), array.compute_read());

	return packed;
#endif
}

PackedByteArray NDArray::to_packed_byte_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to packed");

	PackedByteArray packed;
	packed.resize(static_cast<int64_t>(array.shape[0]));
	fill_c_array_flat(packed.ptrw(), array.compute_read());

	return packed;
#endif
}

PackedInt32Array NDArray::to_packed_int32_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to packed");

	PackedInt32Array packed;
	packed.resize(static_cast<int64_t>(array.shape[0]));
	fill_c_array_flat(packed.ptrw(), array.compute_read());

	return packed;
#endif
}

PackedInt64Array NDArray::to_packed_int64_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 1, {}, "flatten the array before converting to packed");

	PackedInt64Array packed;
	packed.resize(static_cast<int64_t>(array.shape[0]));
	fill_c_array_flat(packed.ptrw(), array.compute_read());

	return packed;
#endif
}

PackedVector2Array NDArray::to_packed_vector2_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 2, {}, "flatten the array before converting to packed");
	ERR_FAIL_COND_V_MSG(array.shape[1] != 2, {}, "final array dimension must be size 2");
	// TODO Handle row major/minor? This still assumes it's normal row-major, i think.

	PackedVector2Array packed;
	packed.resize(static_cast<int64_t>(array.shape[0] * 2));
	fill_c_array_flat(&packed.ptrw()->coord[0], array.compute_read());

	return packed;
#endif
}

PackedVector3Array NDArray::to_packed_vector3_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 3, {}, "flatten the array before converting to packed");
	ERR_FAIL_COND_V_MSG(array.shape[1] != 3, {}, "final array dimension must be size 2");
	// TODO Handle row major/minor? This still assumes it's normal row-major, i think.

	PackedVector3Array packed;
	packed.resize(static_cast<int64_t>(array.shape[0] * 3));
	fill_c_array_flat(&packed.ptrw()->coord[0], array.compute_read());

	return packed;
#endif
}

PackedVector4Array NDArray::to_packed_vector4_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 3, {}, "flatten the array before converting to packed");
	ERR_FAIL_COND_V_MSG(array.shape[1] != 3, {}, "final array dimension must be size 2");
	// TODO Handle row major/minor? This still assumes it's normal row-major, i think.

	PackedVector4Array packed;
	packed.resize(static_cast<int64_t>(array.shape[0] * 3));
	fill_c_array_flat(&packed.ptrw()->components[0], array.compute_read());

	return packed;
#endif
}

PackedColorArray NDArray::to_packed_color_array() const {
#ifdef NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS to enable it.");
#else
	ERR_FAIL_COND_V_MSG(array.dimension() != 3, {}, "flatten the array before converting to packed");
	ERR_FAIL_COND_V_MSG(array.shape[1] != 3, {}, "final array dimension must be size 2");
	// TODO Handle row major/minor? This still assumes it's normal row-major, i think.

	PackedColorArray packed;
	packed.resize(static_cast<int64_t>(array.shape[0] * 3));
	fill_c_array_flat(&packed.ptrw()->components[0], array.compute_read());

	return packed;
#endif
}

TypedArray<NDArray> NDArray::to_godot_array() const {
	ERR_FAIL_COND_V_MSG(array.dimension() == 0, {}, "can't slice a 0-dimension vector");

	auto godot_array = TypedArray<NDArray>();
	const std::size_t size = array.size();
	godot_array.resize(static_cast<int64_t>(size));
	for (std::size_t i = 0; i < size; i++) {
		godot_array[static_cast<int64_t>(i)] = {memnew(NDArray(array.slice({i})))};
	}
	return godot_array;
}

template <typename Visitor, typename... Args>
void map_variants_as_arrays_inplace(Visitor&& visitor, va::VArray& target, const Args&... args) {
    try {
		auto compute_variant = target.compute_write();
        std::forward<Visitor>(visitor)(&compute_variant, variant_as_array(args)...);
    }
    catch (std::runtime_error& error) {
        ERR_FAIL_MSG(error.what());
    }
}

template <typename Visitor, typename VisitorNoaxes, typename... Args>
inline void reduction_inplace(Visitor&& visitor, VisitorNoaxes&& visitor_noaxes, va::VArray& target, const Variant& axes, const Args&... args) {
	try {
		if (axes.get_type() == Variant::NIL) {
			const va::VScalar result = std::forward<VisitorNoaxes>(visitor_noaxes)(variant_as_array(args)...);
			auto compute = target.compute_write();
			va::assign(compute, result);
			return;
		}

		const auto axes_ = variant_to_axes(axes);
		auto compute_variant = target.compute_write();

		std::forward<Visitor>(visitor)(&compute_variant, axes_, variant_as_array(args)...);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_MSG(error.what());
	}
}

#define VARRAY_MAP1(func, varray1) \
	map_variants_as_arrays_inplace([this](va::VArrayTarget target, const va::VArray& varray) {\
        va::func(target, varray);\
    }, this->array, (varray1));\
    return {this}

#define VARRAY_MAP2(func, varray1, varray2) \
	map_variants_as_arrays_inplace([this](va::VArrayTarget target, const va::VArray& a, const va::VArray& b) {\
        va::func(target, a, b);\
    }, this->array, (varray1), (varray2));\
    return {this}

#define VARRAY_MAP3(func, varray1, varray2, varray3) \
	map_variants_as_arrays_inplace([this](va::VArrayTarget target, const va::VArray& a, const va::VArray& b, const va::VArray& c) {\
        va::func(target, a, b, c);\
    }, this->array, (varray1), (varray2), (varray3));\
    return {this}

#define REDUCTION1(func, varray1, axes1) \
	reduction_inplace([this](va::VArrayTarget target, const va::axes_type& axes, const va::VArray& array) {\
		va::func(target, array, axes);\
	}, [this](const va::VArray& array) {\
		return va::func(array);\
	}, this->array, (axes1), (varray1));\
	return {this}

#define REDUCTION2(func, varray1, varray2, axes1) \
	reduction_inplace([this](va::VArrayTarget target, const va::axes_type& axes, const va::VArray& carray1, const va::VArray& carray2) {\
		va::func(target, carray1, carray2, axes);\
	}, [this](const va::VArray& carray1, const va::VArray& carray2) {\
		return va::func(carray1, carray2);\
	}, this->array, (axes1), (varray1), (varray2));\
	return {this}

Ref<NDArray> NDArray::assign_add(const Variant &a, const Variant &b) {
	// godot::UtilityFunctions::print(value);
	VARRAY_MAP2(add, a, b);
}

Ref<NDArray> NDArray::assign_subtract(const Variant& a, const Variant& b) {
	VARRAY_MAP2(subtract, a, b);
}

Ref<NDArray> NDArray::assign_multiply(const Variant& a, const Variant &b) {
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

Ref<NDArray> NDArray::assign_var(const Variant& a, const Variant& axes) {
	REDUCTION1(var, a, axes);
}

Ref<NDArray> NDArray::assign_std(const Variant& a, const Variant& axes) {
	REDUCTION1(std, a, axes);
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

Ref<NDArray> NDArray::assign_dot(const Variant& a, const Variant& b) {
	VARRAY_MAP2(dot, a, b);
}

Ref<NDArray> NDArray::assign_reduce_dot(const Variant& a, const Variant& b, const Variant& axes) {
	REDUCTION2(reduce_dot, a, b, axes);
}

Ref<NDArray> NDArray::assign_matmul(const Variant& a, const Variant& b) {
	VARRAY_MAP2(matmul, a, b);
}

#define CONVERT_TO_SCALAR(type)\
try {\
	return static_cast<type>(array);\
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
