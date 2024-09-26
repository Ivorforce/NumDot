#ifndef NUMDOT_ND_H
#define NUMDOT_ND_H

#include "vatensor/auto_defines.h"
#include <cstdint>                            // for int64_t, uint64_t
#include <godot_cpp/classes/ref.hpp>          // for Ref
#include <godot_cpp/core/binder_common.hpp>   // for VARIANT_ENUM_CAST
#include "godot_cpp/classes/object.hpp"       // for Object
#include "godot_cpp/classes/wrapped.hpp"      // for GDCLASS
#include "godot_cpp/core/class_db.hpp"        // for ClassDB (ptr only), DEFVAL
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "godot_cpp/variant/variant.hpp"      // for Variant
#include "ndarray.h"                          // for NDArray
#include "vatensor/varray.h"                           // for DType


using namespace godot;

class nd : public Object {
	GDCLASS(nd, Object)

protected:
	static void _bind_methods();

public:
	// Godot needs nd::DType here.
	using DType = va::DType;

	nd();
	~nd();

	// Constants.
	static StringName newaxis();
	static StringName ellipsis();

	// Range.
	static Vector4i range(const Variant &start_or_stop = static_cast<int32_t>(0), const Variant &stop = nullptr, const Variant &step = DEFVAL(nullptr));
	static Vector4i from(int32_t start);
	static Vector4i to(int32_t stop);

	// Property access.
	static uint64_t size_of_dtype_in_bytes(DType dtype);

	// Array interpretation.
	static Ref<NDArray> as_array(const Variant &array, DType dtype = DType::DTypeMax);
	static Ref<NDArray> array(const Variant &array, DType dtype = DType::DTypeMax);

	// Array creation.
	static Ref<NDArray> empty(const Variant &shape, DType dtype = DType::Float64);
	static Ref<NDArray> full(const Variant& shape, const Variant& fill_value, DType dtype = DType::Float64);
	static Ref<NDArray> zeros(const Variant &shape, DType dtype = DType::Float64);
	static Ref<NDArray> ones(const Variant &shape, DType dtype = DType::Float64);
	static Ref<NDArray> linspace(const Variant &start, const Variant &stop, int64_t num = 50, bool endpoint = true, DType dtype = DType::DTypeMax);
	static Ref<NDArray> arange(const Variant &start_or_stop = static_cast<int64_t>(0), const Variant &stop = nullptr, const Variant &step = static_cast<int64_t>(1), DType dtype = DType::DTypeMax);

	// Rearrange
	static Ref<NDArray> transpose(const Variant &a, const Variant &permutation);
	static Ref<NDArray> reshape(const Variant &a, const Variant &shape);
	static Ref<NDArray> swapaxes(const Variant& v, int64_t a, int64_t b);
	static Ref<NDArray> moveaxis(const Variant& v, int64_t src, int64_t dst);
	static Ref<NDArray> flip(const Variant& v, int64_t axis);
	static Ref<NDArray> stack(const Variant& v, int64_t axis);
	static Ref<NDArray> unstack(const Variant& v, int64_t axis);

	// Basic math functions.
	static Ref<NDArray> add(const Variant& a, const Variant& b);
	static Ref<NDArray> subtract(const Variant& a, const Variant& b);
	static Ref<NDArray> multiply(const Variant& a, const Variant& b);
	static Ref<NDArray> divide(const Variant& a, const Variant& b);
	static Ref<NDArray> remainder(const Variant& a, const Variant& b);
	static Ref<NDArray> pow(const Variant& a, const Variant& b);

	static Ref<NDArray> minimum(const Variant& a, const Variant& b);
	static Ref<NDArray> maximum(const Variant& a, const Variant& b);
	static Ref<NDArray> clip(const Variant& a, const Variant& min, const Variant& max);

	static Ref<NDArray> sign(const Variant& a);
	static Ref<NDArray> abs(const Variant& a);
	static Ref<NDArray> square(const Variant& a);
	static Ref<NDArray> sqrt(const Variant& a);

	static Ref<NDArray> exp(const Variant& a);
	static Ref<NDArray> log(const Variant& a);

	static Ref<NDArray> rad2deg(const Variant& a);
	static Ref<NDArray> deg2rad(const Variant& a);
	
	// Trigonometric functions.
	static Ref<NDArray> sin(const Variant& a);
	static Ref<NDArray> cos(const Variant& a);
	static Ref<NDArray> tan(const Variant& a);
	static Ref<NDArray> asin(const Variant& a);
	static Ref<NDArray> acos(const Variant& a);
	static Ref<NDArray> atan(const Variant& a);
	static Ref<NDArray> atan2(const Variant& x1, const Variant& x2);

	static Ref<NDArray> sinh(const Variant& a);
	static Ref<NDArray> cosh(const Variant& a);
	static Ref<NDArray> tanh(const Variant& a);
	static Ref<NDArray> asinh(const Variant& a);
	static Ref<NDArray> acosh(const Variant& a);
	static Ref<NDArray> atanh(const Variant& a);

	// Reductions.
	static Ref<NDArray> sum(const Variant& a, const Variant& axes);
	static Ref<NDArray> prod(const Variant& a, const Variant& axes);
	static Ref<NDArray> mean(const Variant& a, const Variant& axes);
	static Ref<NDArray> var(const Variant& a, const Variant& axes);
	static Ref<NDArray> std(const Variant& a, const Variant& axes);
	static Ref<NDArray> max(const Variant& a, const Variant& axes);
	static Ref<NDArray> min(const Variant& a, const Variant& axes);
	static Ref<NDArray> norm(const Variant& a, const Variant& ord, const Variant& axes);

	// Rounding.
	static Ref<NDArray> floor(const Variant& a);
	static Ref<NDArray> ceil(const Variant& a);
	static Ref<NDArray> round(const Variant& a);
	static Ref<NDArray> trunc(const Variant& a);
	static Ref<NDArray> rint(const Variant& a);

	// Comparisons.
	static Ref<NDArray> equal(const Variant& a, const Variant& b);
	static Ref<NDArray> not_equal(const Variant& a, const Variant& b);
	static Ref<NDArray> greater(const Variant& a, const Variant& b);
	static Ref<NDArray> greater_equal(const Variant& a, const Variant& b);
	static Ref<NDArray> less(const Variant& a, const Variant& b);
	static Ref<NDArray> less_equal(const Variant& a, const Variant& b);

	// Logical.
	static Ref<NDArray> logical_and(const Variant& a, const Variant& b);
	static Ref<NDArray> logical_or(const Variant& a, const Variant& b);
	static Ref<NDArray> logical_xor(const Variant& a, const Variant& b);
	static Ref<NDArray> logical_not(const Variant& a);
    static Ref<NDArray> all(const Variant& a, const Variant& axes);
    static Ref<NDArray> any(const Variant& a, const Variant& axes);

	// Linalg.
	static Ref<NDArray> dot(const Variant& a, const Variant& b);
	static Ref<NDArray> reduce_dot(const Variant& a, const Variant& b, const Variant& axes);
	static Ref<NDArray> matmul(const Variant& a, const Variant& b);
};

VARIANT_ENUM_CAST(nd::DType);

#endif
