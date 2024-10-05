#include "nd.hpp"

#include <vatensor/comparison.hpp>            // for equal_to, greater, greate...
#include <vatensor/comparison.hpp>            // for equal_to, greater, greate...
#include <vatensor/linalg.hpp>                // for reduce_dot, dot, matmul
#include <vatensor/logical.hpp>               // for logical_and, logical_not
#include <vatensor/reduce.hpp>                // for all, any, max, mean, median
#include <vatensor/round.hpp>                 // for ceil, floor, nearbyint
#include <vatensor/trigonometry.hpp>          // for acos, acosh, asin, asinh
#include <vatensor/vassign.hpp>               // for assign
#include <vatensor/vmath.hpp>                 // for abs, add, clip, deg2rad
#include <cmath>                            // for double_t, isinf
#include <optional>                         // for optional
#include <stdexcept>                        // for runtime_error
#include <memory>                           // shared_ptr
#include <type_traits>                      // for decay_t
#include <utility>                          // for forward
#include <variant>                          // for visit
#include "gdconvert/conversion_array.hpp"     // for variant_as_array
#include "gdconvert/conversion_ints.hpp"      // for variant_to_axes, variant_...
#include "gdconvert/conversion_slice.hpp"     // for ellipsis, newaxis
#include "godot_cpp/classes/ref.hpp"        // for Ref
#include "godot_cpp/core/error_macros.hpp"  // for ERR_FAIL_V_MSG, ERR_FAIL_...
#include "godot_cpp/core/memory.hpp"        // for _post_initialize, memnew
#include "ndarray.hpp"                        // for NDArray
#include "vatensor/allocate.hpp"              // for full, empty
#include "vatensor/rearrange.hpp"             // for moveaxis, reshape, transpose
#include "vatensor/varray.hpp"                // for VArrayTarget, axes_type
#include "xtensor/xbuilder.hpp"             // for arange, linspace
#include "xtensor/xlayout.hpp"              // for layout_type


using namespace godot;

void nd::_bind_methods() {
	// For the macros, we need to have the values in our namespace.
	using namespace va;
	BIND_ENUM_CONSTANT(Bool);
	BIND_ENUM_CONSTANT(Float32);
	BIND_ENUM_CONSTANT(Float64);
	BIND_ENUM_CONSTANT(Int8);
	BIND_ENUM_CONSTANT(Int16);
	BIND_ENUM_CONSTANT(Int32);
	BIND_ENUM_CONSTANT(Int64);
	BIND_ENUM_CONSTANT(UInt8);
	BIND_ENUM_CONSTANT(UInt16);
	BIND_ENUM_CONSTANT(UInt32);
	BIND_ENUM_CONSTANT(UInt64);

	godot::ClassDB::bind_static_method("nd", D_METHOD("newaxis"), &nd::newaxis);
	godot::ClassDB::bind_static_method("nd", D_METHOD("ellipsis"), &nd::ellipsis);

	godot::ClassDB::bind_static_method("nd", D_METHOD("from", "start"), &nd::from);
	godot::ClassDB::bind_static_method("nd", D_METHOD("to", "stop"), &nd::to);
	godot::ClassDB::bind_static_method("nd", D_METHOD("range", "start_or_stop", "stop", "step"), &nd::range, static_cast<int64_t>(0), DEFVAL(nullptr), DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("size_of_dtype_in_bytes", "dtype"), &nd::size_of_dtype_in_bytes);

	godot::ClassDB::bind_static_method("nd", D_METHOD("as_array", "array", "dtype"), &nd::as_array, DEFVAL(nullptr), DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("array", "array", "dtype"), &nd::array, DEFVAL(nullptr), DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("copy", "array"), &nd::copy);

	godot::ClassDB::bind_static_method("nd", D_METHOD("bool_", "array"), &nd::bool_);
	godot::ClassDB::bind_static_method("nd", D_METHOD("float32", "array"), &nd::float32);
	godot::ClassDB::bind_static_method("nd", D_METHOD("float64", "array"), &nd::float64);
	godot::ClassDB::bind_static_method("nd", D_METHOD("int8", "array"), &nd::int8);
	godot::ClassDB::bind_static_method("nd", D_METHOD("int16", "array"), &nd::int16);
	godot::ClassDB::bind_static_method("nd", D_METHOD("int32", "array"), &nd::int32);
	godot::ClassDB::bind_static_method("nd", D_METHOD("int64", "array"), &nd::int64);
	godot::ClassDB::bind_static_method("nd", D_METHOD("uint8", "array"), &nd::uint8);
	godot::ClassDB::bind_static_method("nd", D_METHOD("uint16", "array"), &nd::uint16);
	godot::ClassDB::bind_static_method("nd", D_METHOD("uint32", "array"), &nd::uint32);
	godot::ClassDB::bind_static_method("nd", D_METHOD("uint64", "array"), &nd::uint64);

	godot::ClassDB::bind_static_method("nd", D_METHOD("empty", "shape", "dtype"), &nd::empty, DEFVAL(nullptr), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("empty_like", "model", "dtype", "shape"), &nd::empty_like, DEFVAL(nullptr), DEFVAL(nd::DType::DTypeMax), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("full", "shape", "fill_value", "dtype"), &nd::full, DEFVAL(nullptr), DEFVAL(nullptr), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("full_like", "model", "fill_value", "dtype", "shape"), &nd::full_like, DEFVAL(nullptr), DEFVAL(nullptr), DEFVAL(nd::DType::DTypeMax), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("zeros", "shape", "dtype"), &nd::zeros, DEFVAL(nullptr), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("ones_like", "model", "dtype", "shape"), &nd::ones_like, DEFVAL(nullptr), DEFVAL(nd::DType::DTypeMax), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("ones", "shape", "dtype"), &nd::ones, DEFVAL(nullptr), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("zeros_like", "model", "dtype", "shape"), &nd::zeros_like, DEFVAL(nullptr), DEFVAL(nd::DType::DTypeMax), DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("eye", "shape", "k", "dtype"), &nd::eye, DEFVAL(nullptr), DEFVAL(0), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("linspace", "start", "stop", "num", "endpoint", "dtype"), &nd::linspace, DEFVAL(0), DEFVAL(nullptr), DEFVAL(50), DEFVAL(true), DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("arange", "start_or_stop", "stop", "step", "dtype"), &nd::arange, DEFVAL(0), DEFVAL(nullptr), DEFVAL(1), DEFVAL(nd::DType::DTypeMax));

	godot::ClassDB::bind_static_method("nd", D_METHOD("transpose", "a", "permutation"), &nd::transpose);
	godot::ClassDB::bind_static_method("nd", D_METHOD("reshape", "a", "shape"), &nd::reshape);
	godot::ClassDB::bind_static_method("nd", D_METHOD("swapaxes", "v", "a", "b"), &nd::swapaxes);
	godot::ClassDB::bind_static_method("nd", D_METHOD("moveaxis", "v", "src", "dst"), &nd::moveaxis);
	godot::ClassDB::bind_static_method("nd", D_METHOD("flip", "v", "axis"), &nd::flip);
	godot::ClassDB::bind_static_method("nd", D_METHOD("stack", "v", "axis"), &nd::stack, DEFVAL(nullptr), 0);
	godot::ClassDB::bind_static_method("nd", D_METHOD("unstack", "v", "axis"), &nd::unstack, DEFVAL(nullptr), 0);

	godot::ClassDB::bind_static_method("nd", D_METHOD("add", "a", "b"), &nd::add);
	godot::ClassDB::bind_static_method("nd", D_METHOD("subtract", "a", "b"), &nd::subtract);
	godot::ClassDB::bind_static_method("nd", D_METHOD("multiply", "a", "b"), &nd::multiply);
	godot::ClassDB::bind_static_method("nd", D_METHOD("divide", "a", "b"), &nd::divide);
	godot::ClassDB::bind_static_method("nd", D_METHOD("remainder", "a", "b"), &nd::remainder);
	godot::ClassDB::bind_static_method("nd", D_METHOD("pow", "a", "b"), &nd::pow);

	godot::ClassDB::bind_static_method("nd", D_METHOD("minimum", "a", "b"), &nd::minimum);
	godot::ClassDB::bind_static_method("nd", D_METHOD("maximum", "a", "b"), &nd::maximum);
	godot::ClassDB::bind_static_method("nd", D_METHOD("clip", "a", "min", "max"), &nd::clip);

	godot::ClassDB::bind_static_method("nd", D_METHOD("sign", "a"), &nd::sign);
	godot::ClassDB::bind_static_method("nd", D_METHOD("abs", "a"), &nd::abs);
	godot::ClassDB::bind_static_method("nd", D_METHOD("square", "a"), &nd::square);
	godot::ClassDB::bind_static_method("nd", D_METHOD("sqrt", "a"), &nd::sqrt);

	godot::ClassDB::bind_static_method("nd", D_METHOD("exp", "a"), &nd::exp);
	godot::ClassDB::bind_static_method("nd", D_METHOD("log", "a"), &nd::log);

	godot::ClassDB::bind_static_method("nd", D_METHOD("rad2deg", "a"), &nd::rad2deg);
	godot::ClassDB::bind_static_method("nd", D_METHOD("deg2rad", "a"), &nd::deg2rad);

	godot::ClassDB::bind_static_method("nd", D_METHOD("sin", "a"), &nd::sin);
	godot::ClassDB::bind_static_method("nd", D_METHOD("cos", "a"), &nd::cos);
	godot::ClassDB::bind_static_method("nd", D_METHOD("tan", "a"), &nd::tan);
	godot::ClassDB::bind_static_method("nd", D_METHOD("asin", "a"), &nd::asin);
	godot::ClassDB::bind_static_method("nd", D_METHOD("acos", "a"), &nd::acos);
	godot::ClassDB::bind_static_method("nd", D_METHOD("atan", "a"), &nd::atan);
	godot::ClassDB::bind_static_method("nd", D_METHOD("atan2", "x1", "x2"), &nd::atan2);

	godot::ClassDB::bind_static_method("nd", D_METHOD("sinh", "a"), &nd::sinh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("cosh", "a"), &nd::cosh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("tanh", "a"), &nd::tanh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("asinh", "a"), &nd::asinh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("acosh", "a"), &nd::acosh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("atanh", "a"), &nd::atanh);

	godot::ClassDB::bind_static_method("nd", D_METHOD("sum", "a", "axes"), &nd::sum, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("prod", "a", "axes"), &nd::prod, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("mean", "a", "axes"), &nd::mean, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("median", "a", "axes"), &nd::median, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("var", "a", "axes"), &nd::var, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("std", "a", "axes"), &nd::std, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("max", "a", "axes"), &nd::max, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("min", "a", "axes"), &nd::min, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("norm", "a", "ord", "axes"), &nd::norm, DEFVAL(nullptr), DEFVAL(2), DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("floor", "a"), &nd::floor);
	godot::ClassDB::bind_static_method("nd", D_METHOD("ceil", "a"), &nd::ceil);
	godot::ClassDB::bind_static_method("nd", D_METHOD("round", "a"), &nd::round);
	godot::ClassDB::bind_static_method("nd", D_METHOD("trunc", "a"), &nd::trunc);
	godot::ClassDB::bind_static_method("nd", D_METHOD("rint", "a"), &nd::rint);

	godot::ClassDB::bind_static_method("nd", D_METHOD("equal", "a", "b"), &nd::equal);
	godot::ClassDB::bind_static_method("nd", D_METHOD("not_equal", "a", "b"), &nd::not_equal);
	godot::ClassDB::bind_static_method("nd", D_METHOD("greater", "a", "b"), &nd::greater);
	godot::ClassDB::bind_static_method("nd", D_METHOD("greater_equal", "a", "b"), &nd::greater_equal);
	godot::ClassDB::bind_static_method("nd", D_METHOD("less", "a", "b"), &nd::less);
	godot::ClassDB::bind_static_method("nd", D_METHOD("less_equal", "a", "b"), &nd::less_equal);

	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_and", "a", "b"), &nd::logical_and);
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_or", "a", "b"), &nd::logical_or);
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_xor", "a", "b"), &nd::logical_xor);
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_not", "a"), &nd::logical_not);
	godot::ClassDB::bind_static_method("nd", D_METHOD("all", "a", "axes"), &nd::all, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("any", "a", "axes"), &nd::any, DEFVAL(nullptr), DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("dot", "a", "b"), &nd::dot);
	godot::ClassDB::bind_static_method("nd", D_METHOD("reduce_dot", "a", "b", "axes"), &nd::reduce_dot, DEFVAL(nullptr), DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("matmul", "a", "b"), &nd::matmul);

	godot::ClassDB::bind_static_method("nd", D_METHOD("default_rng", "seed"), &nd::default_rng, DEFVAL(nullptr));
}

nd::nd() = default;
nd::~nd() = default;

template<typename Visitor, typename... Args>
Ref<NDArray> map_variants_as_arrays(Visitor&& visitor, const Args&... args) {
	try {
		const std::shared_ptr<va::VArray> result = std::forward<Visitor>(visitor)(*variant_as_array(args)...);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

template<typename Visitor, typename... Args>
Ref<NDArray> map_variants_as_arrays_with_target(Visitor&& visitor, const Args&... args) {
	try {
		std::shared_ptr<va::VArray> result;
		std::forward<Visitor>(visitor)(&result, *variant_as_array(args)...);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

template<typename Visitor, typename VisitorNoaxes, typename... Args>
inline Ref<NDArray> reduction(Visitor&& visitor, VisitorNoaxes&& visitor_noaxes, const Variant& axes, const Args&... args) {
	try {
		if (axes.get_type() == Variant::NIL) {
			const auto result = std::forward<VisitorNoaxes>(visitor_noaxes)(*variant_as_array(args)...);

			if constexpr (std::is_same_v<std::decay_t<decltype(result)>, va::VScalar>) {
				return { memnew(NDArray(va::from_scalar_variant(result))) };
			}
			else {
				return { memnew(NDArray(va::from_scalar(result))) };
			}
		}

		const auto axes_ = variant_to_axes(axes);

		std::shared_ptr<va::VArray> result;
		std::forward<Visitor>(visitor)(&result, axes_, *variant_as_array(args)...);

		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

template<typename Visitor>
Ref<NDArray> like_visit(Visitor&& visitor, const Variant& model, nd::DType dtype, const Variant& shape) {
	try {
		va::shape_type shape_used;
		va::DType dtype_used;

		if (dtype != nd::DType::DTypeMax || shape.get_type() != Variant::NIL)
			find_shape_and_dtype(shape_used, dtype_used, model);

		if (dtype != nd::DType::DTypeMax)
			dtype_used = dtype;

		if (shape.get_type() != Variant::NIL)
			shape_used = variant_to_shape(shape);

		return std::forward<Visitor>(visitor)(shape_used, dtype_used);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

#define VARRAY_MAP1(func, varray1) \
	map_variants_as_arrays_with_target([](const va::VArrayTarget target, const va::VArray& varray) {\
        va::func(target, varray);\
    }, (varray1))

#define VARRAY_MAP2(func, varray1, varray2) \
	map_variants_as_arrays_with_target([](const va::VArrayTarget target, const va::VArray& a, const va::VArray& b) {\
        va::func(target, a, b);\
    }, (varray1), (varray2))

#define VARRAY_MAP3(func, varray1, varray2, varray3) \
	map_variants_as_arrays_with_target([](const va::VArrayTarget target, const va::VArray& a, const va::VArray& b, const va::VArray& c) {\
        va::func(target, a, b, c);\
    }, (varray1), (varray2), (varray3))

#define REDUCTION1(func, varray1, axes1) \
	reduction([](const va::VArrayTarget target, const va::axes_type& axes, const va::VArray& array) {\
		va::func(target, array, axes);\
	}, [](const va::VArray& array) { return va::func(array); }, axes, (varray1))

#define REDUCTION2(func, varray1, varray2, axes1) \
	reduction([](const va::VArrayTarget target, const va::axes_type& axes, const va::VArray& carray1, const va::VArray& carray2) {\
		va::func(target, carray1, carray2, axes);\
	}, [](const va::VArray& carray1, const va::VArray& carray2) {\
		return va::func(carray1, carray2);\
	}, axes, (varray1), (varray2))

StringName nd::newaxis() {
	return ::newaxis();
}

StringName nd::ellipsis() {
	return ::ellipsis();
}

Vector4i nd::from(int32_t start) {
	return Vector4i(0b100, start, 0, 0);
}

Vector4i nd::to(int32_t stop) {
	return Vector4i(0b010, 0, stop, 0);
}

Vector4i nd::range(const Variant& start_or_stop, const Variant& stop, const Variant& step) {
	const auto type1 = start_or_stop.get_type();
	const auto type2 = stop.get_type();
	const auto type3 = step.get_type();
	ERR_FAIL_COND_V_MSG(
		(type1 != Variant::NIL && type1 != Variant::INT) || (type2 != Variant::NIL && type2 != Variant::INT) || (type3 != Variant::NIL && type3 != Variant::INT),
		Vector4i(),
		"All arguments to range must be ints or nil."
	);

	const auto mask = (0b100 * (type1 == Variant::INT)) | (0b010 * (type2 == Variant::INT)) | (0b001 * (type3 == Variant::INT));
	if (mask == 0b100) {
		// Special case: nd.range(x)
		// TODO presumably! No real way to check....
		return Vector4i(0b010, 0, static_cast<int32_t>(start_or_stop), 0);
	}

	return Vector4i(
		mask,
		// These default to 0 when type is NIL
		static_cast<int32_t>(start_or_stop),
		static_cast<int32_t>(stop),
		static_cast<int32_t>(step)
	);
}

uint64_t nd::size_of_dtype_in_bytes(const DType dtype) {
	return va::size_of_dtype_in_bytes(dtype);
}

Ref<NDArray> nd::as_array(const Variant& array, const nd::DType dtype) {
	try {
		const auto result = variant_as_array(array, dtype, false);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::array(const Variant& array, nd::DType dtype) {
	try {
		const auto result = variant_as_array(array, dtype, true);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::copy(const Variant& array) {
	return nd::array(array, nd::DType::DTypeMax);
}

Ref<NDArray> nd::bool_(const Variant& array) { return nd::as_array(array, DType::Bool); }
Ref<NDArray> nd::float32(const Variant& array) { return nd::as_array(array, DType::Float32); }
Ref<NDArray> nd::float64(const Variant& array) { return nd::as_array(array, DType::Float64); }
Ref<NDArray> nd::int8(const Variant& array) { return nd::as_array(array, DType::Int8); }
Ref<NDArray> nd::int16(const Variant& array) { return nd::as_array(array, DType::Int16); }
Ref<NDArray> nd::int32(const Variant& array) { return nd::as_array(array, DType::Int32); }
Ref<NDArray> nd::int64(const Variant& array) { return nd::as_array(array, DType::Int64); }
Ref<NDArray> nd::uint8(const Variant& array) { return nd::as_array(array, DType::UInt8); }
Ref<NDArray> nd::uint16(const Variant& array) { return nd::as_array(array, DType::UInt16); }
Ref<NDArray> nd::uint32(const Variant& array) { return nd::as_array(array, DType::UInt32); }
Ref<NDArray> nd::uint64(const Variant& array) { return nd::as_array(array, DType::UInt64); }

Ref<NDArray> nd::empty_like(const Variant& model, nd::DType dtype, const Variant& shape) {
	return like_visit(
		[](va::shape_type& shape, nd::DType& dtype) -> Ref<NDArray> {
			return { memnew(NDArray(va::empty(dtype, shape))) };
		}, model, dtype, shape
	);
}

Ref<NDArray> nd::empty(const Variant& shape, const nd::DType dtype) {
	try {
		const auto shape_array = variant_to_shape(shape);

		return { memnew(NDArray(va::empty(dtype, shape_array))) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> full(const va::shape_type& shape, nd::DType dtype, const Variant& fill_value) {
	switch (fill_value.get_type()) {
		case Variant::BOOL: {
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Bool;
			const auto value = va::scalar_to_dtype(static_cast<bool>(fill_value), dtype);
			return { memnew(NDArray(va::full(value, shape))) };
		}
		case Variant::INT: {
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Int64;
			const auto value = va::scalar_to_dtype(static_cast<int64_t>(fill_value), dtype);
			return { memnew(NDArray(va::full(value, shape))) };
		}
		case Variant::FLOAT: {
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Float64;
			const auto value = va::scalar_to_dtype(static_cast<double_t>(fill_value), dtype);
			return { memnew(NDArray(va::full(value, shape))) };
		}
		default: {
			std::shared_ptr<va::VArray> result = va::empty(dtype, shape);
			result->prepare_write();
			va::assign(result->write.value(), variant_as_array(fill_value)->read);
			return { memnew(NDArray(result)) };
		}
	}

	ERR_FAIL_V_MSG({}, "The fill value must be a number literal (for now).");
}

Ref<NDArray> nd::full(const Variant& shape, const Variant& fill_value, nd::DType dtype) {
	try {
		const auto shape_array = variant_to_shape(shape);
		return ::full(shape_array, dtype, fill_value);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::full_like(const Variant& model, const Variant& fill_value, nd::DType dtype, const Variant& shape) {
	return like_visit(
		[fill_value](va::shape_type& shape, nd::DType dtype) -> Ref<NDArray> {
			return ::full(shape, dtype, fill_value);
		}, model, dtype, shape
	);
}

Ref<NDArray> nd::zeros(const Variant& shape, const nd::DType dtype) {
	return full(shape, 0, dtype);
}

Ref<NDArray> nd::zeros_like(const Variant& model, nd::DType dtype, const Variant& shape) {
	return full_like(model, 0, dtype, shape);
}

auto nd::ones(const Variant& shape, const nd::DType dtype) -> Ref<NDArray> {
	return full(shape, 1, dtype);
}

Ref<NDArray> nd::ones_like(const Variant& model, nd::DType dtype, const Variant& shape) {
	return full_like(model, 1, dtype, shape);
}

Ref<NDArray> nd::eye(const Variant& shape, int64_t k, nd::DType dtype) {
	try {
		va::shape_type used_shape = shape.get_type() == Variant::INT
		                            ? va::shape_type { static_cast<va::size_type>(static_cast<int64_t>(shape)), static_cast<va::size_type>(static_cast<int64_t>(shape)) }
		                            : variant_to_shape(shape);

		auto result = va::eye(dtype, used_shape, k);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::linspace(const Variant& start, const Variant& stop, const int64_t num, const bool endpoint, DType dtype) {
#ifdef NUMDOT_DISABLE_ALLOCATION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ALLOCATION_FUNCTIONS to enable it.");
#else
	if (dtype == DType::DTypeMax) {
		dtype = start.get_type() == Variant::FLOAT || stop.get_type() == Variant::FLOAT
		        ? nd::DType::Float64
		        : nd::DType::Float32;
	}

	try {
		const auto result = std::visit(
			[start, stop, num, endpoint](auto t) -> std::shared_ptr<va::VArray> {
				using T = std::decay_t<decltype(t)>;

				if constexpr (std::is_floating_point_v<T>) {
					auto store = va::make_store<T>(xt::linspace(static_cast<double_t>(start), static_cast<double_t>(stop), num, endpoint));
					return va::from_store(store);
				}
				else {
					auto store = va::make_store<T>(xt::linspace(static_cast<int64_t>(start), static_cast<int64_t>(stop), num, endpoint));
					return va::from_store(store);
				}
			}, va::dtype_to_variant(dtype)
		);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
#endif
}

Ref<NDArray> nd::arange(const Variant& start_or_stop, const Variant& stop, const Variant& step, DType dtype) {
#ifdef NUMDOT_DISABLE_ALLOCATION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ALLOCATION_FUNCTIONS to enable it.");
#else
	if (dtype == DType::DTypeMax) {
		dtype = start_or_stop.get_type() == Variant::FLOAT || stop.get_type() == Variant::FLOAT || step.get_type() == Variant::FLOAT
		        ? nd::DType::Float64
		        : nd::DType::Int64;
	}
	static const Variant zero = 0;
	const Variant& start_ = stop.get_type() == Variant::NIL ? zero : start_or_stop;
	const Variant& stop_ = stop.get_type() == Variant::NIL ? start_or_stop : nullptr;
	const Variant& step_ = step;

	try {
		const auto result = std::visit(
			[start_, stop_, step_](auto t) -> std::shared_ptr<va::VArray> {
				using T = std::decay_t<decltype(t)>;

				if constexpr (std::is_floating_point_v<T>) {
					const auto store = va::make_store<T>(xt::arange(static_cast<double_t>(start_), static_cast<double_t>(stop_), static_cast<double_t>(step_)));
					return va::from_store(store);
				}
				else {
					const auto store = va::make_store<T>(xt::arange(static_cast<int64_t>(start_), static_cast<int64_t>(stop_), static_cast<int64_t>(step_)));
					return va::from_store(store);
				}
			}, va::dtype_to_variant(dtype)
		);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
#endif
}

Ref<NDArray> nd::transpose(const Variant& a, const Variant& permutation) {
	try {
		std::shared_ptr<va::VArray> a_ = variant_as_array(a);
		// TODO It's not exactly a shape, but 'int array' is close enough.
		//  We should probably decouple them when we add better shape checks.
		const auto permutation_ = variant_to_axes(permutation);

		return { memnew(NDArray(va::transpose(*a_, permutation_))) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::reshape(const Variant& a, const Variant& shape) {
	try {
		std::shared_ptr<va::VArray> a_ = variant_as_array(a);
		// TODO It's not exactly a shape, but 'int array' is close enough.
		//  We should probably decouple them when we add better shape checks.
		const auto new_shape_ = variant_to_axes(shape);

		return { memnew(NDArray(va::reshape(*a_, new_shape_))) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::swapaxes(const Variant& v, const int64_t a, const int64_t b) {
	return map_variants_as_arrays(
		[a, b](const va::VArray& v) {
			return va::swapaxes(v, a, b);
		}, v
	);
}

Ref<NDArray> nd::moveaxis(const Variant& v, int64_t src, int64_t dst) {
	return map_variants_as_arrays(
		[src, dst](const va::VArray& v) {
			return va::moveaxis(v, src, dst);
		}, v
	);
}

Ref<NDArray> nd::flip(const Variant& v, int64_t axis) {
	return map_variants_as_arrays(
		[axis](const va::VArray& v) {
			return va::flip(v, axis);
		}, v
	);
}

Ref<NDArray> nd::stack(const Variant& v, int64_t axis) {
	return moveaxis(v, 0, axis);
}

Ref<NDArray> nd::unstack(const Variant& v, int64_t axis) {
	return moveaxis(v, axis, 0);
}

Ref<NDArray> nd::add(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(add, a, b);
}

Ref<NDArray> nd::subtract(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(subtract, a, b);
}

Ref<NDArray> nd::multiply(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(multiply, a, b);
}

Ref<NDArray> nd::divide(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(divide, a, b);
}

Ref<NDArray> nd::remainder(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(remainder, a, b);
}

Ref<NDArray> nd::pow(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(pow, a, b);
}

Ref<NDArray> nd::minimum(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(minimum, a, b);
}

Ref<NDArray> nd::maximum(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(maximum, a, b);
}

Ref<NDArray> nd::clip(const Variant& a, const Variant& min, const Variant& max) {
	return VARRAY_MAP3(clip, a, min, max);
}

Ref<NDArray> nd::sign(const Variant& a) {
	return VARRAY_MAP1(sign, a);
}

Ref<NDArray> nd::abs(const Variant& a) {
	return VARRAY_MAP1(abs, a);
}

Ref<NDArray> nd::square(const Variant& a) {
	return VARRAY_MAP1(square, a);
}

Ref<NDArray> nd::sqrt(const Variant& a) {
	return VARRAY_MAP1(sqrt, a);
}

Ref<NDArray> nd::exp(const Variant& a) {
	return VARRAY_MAP1(exp, a);
}

Ref<NDArray> nd::log(const Variant& a) {
	return VARRAY_MAP1(log, a);
}

Ref<NDArray> nd::rad2deg(const Variant& a) {
	return VARRAY_MAP1(rad2deg, a);
}

Ref<NDArray> nd::deg2rad(const Variant& a) {
	return VARRAY_MAP1(deg2rad, a);
}

Ref<NDArray> nd::sin(const Variant& a) {
	return VARRAY_MAP1(sin, a);
}

Ref<NDArray> nd::cos(const Variant& a) {
	return VARRAY_MAP1(cos, a);
}

Ref<NDArray> nd::tan(const Variant& a) {
	return VARRAY_MAP1(tan, a);
}

Ref<NDArray> nd::asin(const Variant& a) {
	return VARRAY_MAP1(asin, a);
}

Ref<NDArray> nd::acos(const Variant& a) {
	return VARRAY_MAP1(acos, a);
}

Ref<NDArray> nd::atan(const Variant& a) {
	return VARRAY_MAP1(atan, a);
}

Ref<NDArray> nd::atan2(const Variant& x1, const Variant& x2) {
	return VARRAY_MAP2(atan2, x1, x2);
}

Ref<NDArray> nd::sinh(const Variant& a) {
	return VARRAY_MAP1(sinh, a);
}

Ref<NDArray> nd::cosh(const Variant& a) {
	return VARRAY_MAP1(cosh, a);
}

Ref<NDArray> nd::tanh(const Variant& a) {
	return VARRAY_MAP1(tanh, a);
}

Ref<NDArray> nd::asinh(const Variant& a) {
	return VARRAY_MAP1(asinh, a);
}

Ref<NDArray> nd::acosh(const Variant& a) {
	return VARRAY_MAP1(acosh, a);
}

Ref<NDArray> nd::atanh(const Variant& a) {
	return VARRAY_MAP1(atanh, a);
}

Ref<NDArray> nd::sum(const Variant& a, const Variant& axes) {
	return REDUCTION1(sum, a, axes);
}

Ref<NDArray> nd::prod(const Variant& a, const Variant& axes) {
	return REDUCTION1(prod, a, axes);
}

Ref<NDArray> nd::mean(const Variant& a, const Variant& axes) {
	return REDUCTION1(mean, a, axes);
}

Ref<NDArray> nd::median(const Variant& a, const Variant& axes) {
	return REDUCTION1(median, a, axes);
}

Ref<NDArray> nd::var(const Variant& a, const Variant& axes) {
	return REDUCTION1(var, a, axes);
}

Ref<NDArray> nd::std(const Variant& a, const Variant& axes) {
	return REDUCTION1(std, a, axes);
}

Ref<NDArray> nd::max(const Variant& a, const Variant& axes) {
	return REDUCTION1(max, a, axes);
}

Ref<NDArray> nd::min(const Variant& a, const Variant& axes) {
	return REDUCTION1(min, a, axes);
}

Ref<NDArray> nd::norm(const Variant& a, const Variant& ord, const Variant& axes) {
	switch (ord.get_type()) {
		case Variant::INT:
			switch (static_cast<int64_t>(ord)) {
				case 0:
					return REDUCTION1(norm_l0, a, axes);
				case 1:
					return REDUCTION1(norm_l1, a, axes);
				case 2:
					return REDUCTION1(norm_l2, a, axes);
				default:
					break;
			}
		case Variant::FLOAT:
			if (std::isinf(static_cast<double_t>(ord))) {
				return REDUCTION1(norm_linf, a, axes);
			}
		default:
			break;
	}

	ERR_FAIL_V_MSG({}, "This norm is currently not supported");
}

Ref<NDArray> nd::floor(const Variant& a) {
	return VARRAY_MAP1(floor, a);
}

Ref<NDArray> nd::ceil(const Variant& a) {
	return VARRAY_MAP1(ceil, a);
}

Ref<NDArray> nd::round(const Variant& a) {
	return VARRAY_MAP1(round, a);
}

Ref<NDArray> nd::trunc(const Variant& a) {
	return VARRAY_MAP1(trunc, a);
}

Ref<NDArray> nd::rint(const Variant& a) {
	// Actually uses nearbyint because rint can throw, which is undesirable in our case, and unlike numpy's behavior.
	return VARRAY_MAP1(nearbyint, a);
}

Ref<NDArray> nd::equal(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(equal_to, a, b);
}

Ref<NDArray> nd::not_equal(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(not_equal_to, a, b);
}

Ref<NDArray> nd::greater(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(greater, a, b);
}

Ref<NDArray> nd::greater_equal(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(greater_equal, a, b);
}

Ref<NDArray> nd::less(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(less, a, b);
}

Ref<NDArray> nd::less_equal(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(less_equal, a, b);
}

Ref<NDArray> nd::logical_and(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(logical_and, a, b);
}

Ref<NDArray> nd::logical_or(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(logical_or, a, b);
}

Ref<NDArray> nd::logical_xor(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(logical_xor, a, b);
}

Ref<NDArray> nd::logical_not(const Variant& a) {
	return VARRAY_MAP1(logical_not, a);
}

Ref<NDArray> nd::all(const Variant& a, const Variant& axes) {
	return REDUCTION1(all, a, axes);
}

Ref<NDArray> nd::any(const Variant& a, const Variant& axes) {
	return REDUCTION1(any, a, axes);
}

Ref<NDArray> nd::dot(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(dot, a, b);
}

Ref<NDArray> nd::reduce_dot(const Variant& a, const Variant& b, const Variant& axes) {
	return REDUCTION2(reduce_dot, a, b, axes);
}

Ref<NDArray> nd::matmul(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(matmul, a, b);
}

Ref<NDRandomGenerator> nd::default_rng(const Variant& seed) {
	switch (seed.get_type()) {
		case Variant::NIL:
			return { memnew(NDRandomGenerator()) };
		case Variant::INT:
			return { memnew(NDRandomGenerator(va::random::VRandomEngine(static_cast<uint64_t>(seed)))) };
		default: ERR_FAIL_V_MSG({}, "The given variant could not be converted to a seed.");
	}
}
