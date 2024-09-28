#ifndef NUMDOT_NDARRAY_H
#define NUMDOT_NDARRAY_H

#ifdef WIN32
#include <windows.h>
#endif

#include "vatensor/auto_defines.h"
#include <cmath>                                      // for double_t
#include <cstdint>                                    // for uint64_t, int64_t
#include <godot_cpp/classes/ref_counted.hpp>           // for RefCounted
#include <godot_cpp/variant/variant.hpp>               // for Variant
#include <utility>                                     // for move
#include "gdextension_interface.h"                     // for GDExtensionCal...
#include "godot_cpp/classes/ref.hpp"                   // for Ref
#include "godot_cpp/classes/wrapped.hpp"               // for GDCLASS
#include "godot_cpp/variant/array.hpp"                 // for Array
#include "godot_cpp/variant/packed_byte_array.hpp"     // for PackedByteArray
#include "godot_cpp/variant/packed_float32_array.hpp"  // for PackedFloat32A...
#include "godot_cpp/variant/packed_float64_array.hpp"  // for PackedFloat64A...
#include "godot_cpp/variant/packed_int32_array.hpp"    // for PackedInt32Array
#include "godot_cpp/variant/packed_int64_array.hpp"    // for PackedInt64Array
#include "godot_cpp/variant/string.hpp"                // for String
#include "godot_cpp/variant/color.hpp"                 // for Color
#include "godot_cpp/variant/packed_color_array.hpp"    // for PackedColorArray
#include "godot_cpp/variant/packed_vector2_array.hpp"  // for PackedVector2A...
#include "godot_cpp/variant/packed_vector3_array.hpp"  // for PackedVector3A...
#include "godot_cpp/variant/packed_vector4_array.hpp"  // for PackedVector4A...
#include "godot_cpp/variant/typed_array.hpp"           // for TypedArray
#include "godot_cpp/variant/vector2.hpp"               // for Vector2
#include "godot_cpp/variant/vector2i.hpp"              // for Vector2i
#include "godot_cpp/variant/vector3.hpp"               // for Vector3
#include "godot_cpp/variant/vector3i.hpp"              // for Vector3i
#include "godot_cpp/variant/vector4.hpp"               // for Vector4
#include "godot_cpp/variant/vector4i.hpp"              // for Vector4i
#include "vatensor/varray.h"                                    // for DType, VArray
namespace godot { class ClassDB; }

using namespace godot;

class NDArray : public RefCounted {
	GDCLASS(NDArray, RefCounted)

private:

protected:
	static void _bind_methods();
	String _to_string() const;

public:
	// We need a shared pointer because things like asarray can return either a view or an array
	va::VArray array;

	NDArray();
	explicit NDArray(va::VArray array) : array(std::move(array)) {};
	~NDArray() override;

	[[nodiscard]] va::DType dtype() const;
	[[nodiscard]] PackedInt64Array shape() const;
	[[nodiscard]] uint64_t size() const;
	[[nodiscard]] uint64_t array_size_in_bytes() const;
	[[nodiscard]] uint64_t ndim() const;

	Variant _iter_init(const Array &p_iter);
	Variant _iter_next(const Array &p_iter);
	Variant _iter_get(const Variant &p_iter);

	// Subscript not available, i think. See object's set_bind / get_bind:
	// I think godot assumes that all [] accesses are keypaths.
	// https://github.com/godotengine/godot/blob/514c564a8c855d798ec6b5a52860e5bca8d57bc9/core/object/object.h#L643
	void set(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	Ref<NDArray> get(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	bool get_bool(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	int64_t get_int(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	double_t get_float(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);

	[[nodiscard]] Variant as_type(va::DType dtype) const;

	[[nodiscard]] bool to_bool() const;
	[[nodiscard]] int64_t to_int() const;
	[[nodiscard]] double_t to_float() const;

    [[nodiscard]] Vector2 to_vector2() const;
    [[nodiscard]] Vector3 to_vector3() const;
    [[nodiscard]] Vector4 to_vector4() const;
    [[nodiscard]] Vector2i to_vector2i() const;
    [[nodiscard]] Vector3i to_vector3i() const;
    [[nodiscard]] Vector4i to_vector4i() const;
    [[nodiscard]] Color to_color() const;

	[[nodiscard]] PackedFloat32Array to_packed_float32_array() const;
	[[nodiscard]] PackedFloat64Array to_packed_float64_array() const;
	[[nodiscard]] PackedByteArray to_packed_byte_array() const;
	[[nodiscard]] PackedInt32Array to_packed_int32_array() const;
	[[nodiscard]] PackedInt64Array to_packed_int64_array() const;
	[[nodiscard]] PackedVector2Array to_packed_vector2_array() const;
	[[nodiscard]] PackedVector3Array to_packed_vector3_array() const;
	[[nodiscard]] PackedVector4Array to_packed_vector4_array() const;
	[[nodiscard]] PackedColorArray to_packed_color_array() const;

    [[nodiscard]] TypedArray<NDArray> to_godot_array() const;

	// Basic math functions.
	Ref<NDArray> assign_add(const Variant &a, const Variant &b);
	Ref<NDArray> assign_subtract(const Variant& a, const Variant& b);
	Ref<NDArray> assign_multiply(const Variant& a, const Variant &b);
	Ref<NDArray> assign_divide(const Variant& a, const Variant& b);
	Ref<NDArray> assign_remainder(const Variant& a, const Variant& b);
	Ref<NDArray> assign_pow(const Variant& a, const Variant& b);

	Ref<NDArray> assign_minimum(const Variant& a, const Variant& b);
	Ref<NDArray> assign_maximum(const Variant& a, const Variant& b);
	Ref<NDArray> assign_clip(const Variant& a, const Variant& min, const Variant& max);

	Ref<NDArray> assign_sign(const Variant& a);
	Ref<NDArray> assign_abs(const Variant& a);
	Ref<NDArray> assign_square(const Variant& a);
	Ref<NDArray> assign_sqrt(const Variant& a);

	Ref<NDArray> assign_exp(const Variant& a);
	Ref<NDArray> assign_log(const Variant& a);

	Ref<NDArray> assign_rad2deg(const Variant& a);
	Ref<NDArray> assign_deg2rad(const Variant& a);
	
	// Trigonometric functions.
	Ref<NDArray> assign_sin(const Variant& a);
	Ref<NDArray> assign_cos(const Variant& a);
	Ref<NDArray> assign_tan(const Variant& a);
	Ref<NDArray> assign_asin(const Variant& a);
	Ref<NDArray> assign_acos(const Variant& a);
	Ref<NDArray> assign_atan(const Variant& a);
	Ref<NDArray> assign_atan2(const Variant& x1, const Variant& x2);

	Ref<NDArray> assign_sinh(const Variant& a);
	Ref<NDArray> assign_cosh(const Variant& a);
	Ref<NDArray> assign_tanh(const Variant& a);
	Ref<NDArray> assign_asinh(const Variant& a);
	Ref<NDArray> assign_acosh(const Variant& a);
	Ref<NDArray> assign_atanh(const Variant& a);

	// Reductions.
	Ref<NDArray> assign_sum(const Variant& a, const Variant& axes);
	Ref<NDArray> assign_prod(const Variant& a, const Variant& axes);
	Ref<NDArray> assign_mean(const Variant& a, const Variant& axes);
	Ref<NDArray> assign_var(const Variant& a, const Variant& axes);
	Ref<NDArray> assign_std(const Variant& a, const Variant& axes);
	Ref<NDArray> assign_max(const Variant& a, const Variant& axes);
	Ref<NDArray> assign_min(const Variant& a, const Variant& axes);
	Ref<NDArray> assign_norm(const Variant& a, const Variant& ord, const Variant& axes);

	// Rounding.
	Ref<NDArray> assign_floor(const Variant& a);
	Ref<NDArray> assign_ceil(const Variant& a);
	Ref<NDArray> assign_round(const Variant& a);
	Ref<NDArray> assign_trunc(const Variant& a);
	Ref<NDArray> assign_rint(const Variant& a);

	// Comparisons.
	Ref<NDArray> assign_equal(const Variant& a, const Variant& b);
	Ref<NDArray> assign_not_equal(const Variant& a, const Variant& b);
	Ref<NDArray> assign_greater(const Variant& a, const Variant& b);
	Ref<NDArray> assign_greater_equal(const Variant& a, const Variant& b);
	Ref<NDArray> assign_less(const Variant& a, const Variant& b);
	Ref<NDArray> assign_less_equal(const Variant& a, const Variant& b);

	// Logical.
	Ref<NDArray> assign_logical_and(const Variant& a, const Variant& b);
	Ref<NDArray> assign_logical_or(const Variant& a, const Variant& b);
	Ref<NDArray> assign_logical_xor(const Variant& a, const Variant& b);
	Ref<NDArray> assign_logical_not(const Variant& a);
    Ref<NDArray> assign_all(const Variant& a, const Variant& axes);
    Ref<NDArray> assign_any(const Variant& a, const Variant& axes);

	// Linalg.
	Ref<NDArray> assign_dot(const Variant& a, const Variant& b);
	Ref<NDArray> assign_reduce_dot(const Variant& a, const Variant& b, const Variant& axes);
	Ref<NDArray> assign_matmul(const Variant& a, const Variant& b);

	// Conversion to other types.
	explicit operator bool() const;
	explicit operator int64_t() const;
	explicit operator int32_t() const;
	explicit operator int16_t() const;
	explicit operator int8_t() const;
	explicit operator uint64_t() const;
	explicit operator uint32_t() const;
	explicit operator uint16_t() const;
	explicit operator uint8_t() const;
	explicit operator double() const;
	explicit operator float() const;
};

#endif
