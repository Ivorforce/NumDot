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
	
	// Subscript not available, i think. See object's set_bind / get_bind:
	// I think godot assumes that all [] accesses are keypaths.
	// https://github.com/godotengine/godot/blob/514c564a8c855d798ec6b5a52860e5bca8d57bc9/core/object/object.h#L643
	void set(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	Ref<NDArray> get(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	double_t get_float(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	int64_t get_int(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);

	[[nodiscard]] Variant as_type(va::DType dtype) const;

	[[nodiscard]] double_t to_float() const;
	[[nodiscard]] int64_t to_int() const;

	[[nodiscard]] PackedFloat32Array to_packed_float32_array() const;
	[[nodiscard]] PackedFloat64Array to_packed_float64_array() const;
	[[nodiscard]] PackedByteArray to_packed_byte_array() const;
	[[nodiscard]] PackedInt32Array to_packed_int32_array() const;
	[[nodiscard]] PackedInt64Array to_packed_int64_array() const;

    [[nodiscard]] Array to_godot_array() const;

	// Basic math functions.
	Ref<NDArray> assign_add(Variant a, Variant b);
	Ref<NDArray> assign_subtract(Variant a, Variant b);
	Ref<NDArray> assign_multiply(Variant a, Variant b);
	Ref<NDArray> assign_divide(Variant a, Variant b);
	Ref<NDArray> assign_remainder(Variant a, Variant b);
	Ref<NDArray> assign_pow(Variant a, Variant b);

	Ref<NDArray> assign_minimum(Variant a, Variant b);
	Ref<NDArray> assign_maximum(Variant a, Variant b);
	Ref<NDArray> assign_clip(Variant a, Variant min, Variant max);

	Ref<NDArray> assign_sign(Variant a);
	Ref<NDArray> assign_abs(Variant a);
	Ref<NDArray> assign_square(Variant a);
	Ref<NDArray> assign_sqrt(Variant a);

	Ref<NDArray> assign_exp(Variant a);
	Ref<NDArray> assign_log(Variant a);

	Ref<NDArray> assign_rad2deg(Variant a);
	Ref<NDArray> assign_deg2rad(Variant a);
	
	// Trigonometric functions.
	Ref<NDArray> assign_sin(Variant a);
	Ref<NDArray> assign_cos(Variant a);
	Ref<NDArray> assign_tan(Variant a);
	Ref<NDArray> assign_asin(Variant a);
	Ref<NDArray> assign_acos(Variant a);
	Ref<NDArray> assign_atan(Variant a);
	Ref<NDArray> assign_atan2(Variant x1, Variant x2);

	Ref<NDArray> assign_sinh(Variant a);
	Ref<NDArray> assign_cosh(Variant a);
	Ref<NDArray> assign_tanh(Variant a);
	Ref<NDArray> assign_asinh(Variant a);
	Ref<NDArray> assign_acosh(Variant a);
	Ref<NDArray> assign_atanh(Variant a);

	// Reductions.
	Ref<NDArray> assign_sum(Variant a, Variant axes);
	Ref<NDArray> assign_prod(Variant a, Variant axes);
	Ref<NDArray> assign_mean(Variant a, Variant axes);
	Ref<NDArray> assign_var(Variant a, Variant axes);
	Ref<NDArray> assign_std(Variant a, Variant axes);
	Ref<NDArray> assign_max(Variant a, Variant axes);
	Ref<NDArray> assign_min(Variant a, Variant axes);
	Ref<NDArray> assign_norm(Variant a, Variant ord, Variant axes);

	// Rounding.
	Ref<NDArray> assign_floor(Variant a);
	Ref<NDArray> assign_ceil(Variant a);
	Ref<NDArray> assign_round(Variant a);
	Ref<NDArray> assign_trunc(Variant a);
	Ref<NDArray> assign_rint(Variant a);

	// Comparisons.
	Ref<NDArray> assign_equal(Variant a, Variant b);
	Ref<NDArray> assign_not_equal(Variant a, Variant b);
	Ref<NDArray> assign_greater(Variant a, Variant b);
	Ref<NDArray> assign_greater_equal(Variant a, Variant b);
	Ref<NDArray> assign_less(Variant a, Variant b);
	Ref<NDArray> assign_less_equal(Variant a, Variant b);

	// Logical.
	Ref<NDArray> assign_logical_and(Variant a, Variant b);
	Ref<NDArray> assign_logical_or(Variant a, Variant b);
	Ref<NDArray> assign_logical_xor(Variant a, Variant b);
	Ref<NDArray> assign_logical_not(Variant a);
    Ref<NDArray> assign_all(Variant a, Variant axes);
    Ref<NDArray> assign_any(Variant a, Variant axes);

	// Linalg.
	Ref<NDArray> assign_dot(Variant a, Variant b, Variant axes);
};

#endif
