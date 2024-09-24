#include "nd.h"

#include <vatensor/comparison.h>            // for equal_to, greater, greate...
#include <vatensor/logical.h>               // for logical_and, logical_not
#include <vatensor/reduce.h>                // for max, mean, min, prod, std
#include <vatensor/round.h>                 // for ceil, floor, nearbyint
#include <vatensor/trigonometry.h>          // for acos, acosh, asin, asinh
#include <vatensor/vmath.h>                 // for abs, add, deg2rad, divide
#include <cmath>                            // for double_t
#include <cstddef>                          // for ptrdiff_t, size_t
#include <functional>                       // for function
#include <memory>                           // for make_shared
#include <optional>                         // for optional
#include <stdexcept>                        // for runtime_error
#include <type_traits>                      // for decay_t
#include <utility>                          // for move
#include <variant>                          // for visit, variant
#include <vector>                           // for vector
#include <vatensor/linalg.h>
#include "gdconvert/conversion_array.h"     // for variant_as_array
#include "gdconvert/conversion_axes.h"      // for variant_to_axes
#include "gdconvert/conversion_range.h"     // for to_range_part
#include "gdconvert/conversion_shape.h"     // for variant_as_shape
#include "gdconvert/conversion_slice.h"     // for ellipsis, newaxis
#include "godot_cpp/classes/ref.hpp"        // for Ref
#include "godot_cpp/core/error_macros.hpp"  // for ERR_FAIL_V_MSG
#include "godot_cpp/core/memory.hpp"        // for _post_initialize, memnew
#include "ndarray.h"                        // for NDArray
#include "ndrange.h"                        // for NDRange
#include "vatensor/allocate.h"              // for empty, full, copy_as_dtype
#include "vatensor/rearrange.h"             // for reshape, transpose, flip
#include "vatensor/varray.h"                // for VArrayTarget, DType, VArray
#include "xtensor/xbuilder.hpp"             // for arange, linspace
#include "xtensor/xlayout.hpp"              // for layout_type
#include "xtensor/xslice.hpp"               // for xtuph
#include "xtensor/xtensor_forward.hpp"      // for xarray


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

	godot::ClassDB::bind_static_method("nd", D_METHOD("empty", "shape", "dtype"), &nd::empty, DEFVAL(nullptr), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("full", "shape", "fill_value", "dtype"), &nd::full, DEFVAL(nullptr), DEFVAL(nullptr), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("zeros", "shape", "dtype"), &nd::zeros, DEFVAL(nullptr), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("ones", "shape", "dtype"), &nd::ones, DEFVAL(nullptr), DEFVAL(nd::DType::Float64));
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
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_not", "a"), &nd::logical_not);
    godot::ClassDB::bind_static_method("nd", D_METHOD("all", "a", "axes"), &nd::all, DEFVAL(nullptr), DEFVAL(nullptr));
    godot::ClassDB::bind_static_method("nd", D_METHOD("any", "a", "axes"), &nd::any, DEFVAL(nullptr), DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("dot", "a", "b"), &nd::dot);
	godot::ClassDB::bind_static_method("nd", D_METHOD("reduce_dot", "a", "b", "axes"), &nd::reduce_dot, DEFVAL(nullptr), DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("matmul", "a", "b"), &nd::matmul);
}

nd::nd() = default;
nd::~nd() = default;

template <typename Visitor, typename... Args>
Ref<NDArray> map_variants_as_arrays(Visitor visitor, Args... args) {
	try {
		const auto result = visitor(variant_as_array(args)...);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

template <typename Visitor, typename... Args>
Ref<NDArray> map_variants_as_arrays_with_target(Visitor visitor, Args... args) {
	try {
		std::optional<va::VArray> result;
		visitor(&result, variant_as_array(args)...);
		return { memnew(NDArray(result.value())) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

template <typename Visitor, typename VisitorNoaxes, typename... Args>
inline Ref<NDArray> reduction(Visitor visitor, VisitorNoaxes visitor_noaxes, Variant& axes, Args&... args) {
	try {
		if (axes.get_type() == Variant::NIL) {
			const auto result = visitor_noaxes(variant_as_array(args)...);

			if constexpr (std::is_same_v<std::decay_t<decltype(result)>, va::VConstant>) {
				return { memnew(NDArray(va::from_constant_variant(result))) };
			}
			else {
				return { memnew(NDArray(va::from_constant(result))) };
			}
		}

		const auto axes_ = variant_to_axes(axes);

		std::optional<va::VArray> result;
		visitor(&result, axes_, variant_as_array(args)...);

		return {memnew(NDArray(result.value()))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

#define UNARY_MAP(func, varray1) \
	map_variants_as_arrays_with_target([](const va::VArrayTarget target, const va::VArray& varray) {\
        va::func(target, varray);\
    }, (varray1))

#define BINARY_MAP(func, varray1, varray2) \
	map_variants_as_arrays_with_target([](const va::VArrayTarget target, const va::VArray& a, const va::VArray& b) {\
        va::func(target, a, b);\
    }, (varray1), (varray2))

#define TERNARY_MAP(func, varray1, varray2, varray3) \
	map_variants_as_arrays_with_target([](const va::VArrayTarget target, const va::VArray& a, const va::VArray& b, const va::VArray& c) {\
        va::func(target, a, b, c);\
    }, (varray1), (varray2), (varray3))

#define REDUCTION1(func, varray1, axes1) \
	reduction([](const va::VArrayTarget target, const va::Axes& axes, const va::VArray& array) {\
		va::func(target, array, axes);\
	}, [](const va::VArray& array) { return va::func(array); }, axes, (varray1))

#define REDUCTION2(func, varray1, varray2, axes1) \
	reduction([](const va::VArrayTarget target, const va::Axes& axes, const va::VArray& carray1, const va::VArray& carray2) {\
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

Ref<NDRange> nd::from(int64_t start) {
	return {memnew(NDRange(
		static_cast<std::ptrdiff_t>(start),
		xt::placeholders::xtuph{},
		xt::placeholders::xtuph{}
	))};
}

Ref<NDRange> nd::to(int64_t stop) {
	return {memnew(NDRange(
		xt::placeholders::xtuph{},
		static_cast<std::ptrdiff_t>(stop),
		xt::placeholders::xtuph{}
	))};
}

Ref<NDRange> nd::range(Variant start_or_stop, Variant stop, Variant step) {
	try {
		if (stop.get_type() == Variant::Type::NIL && step.get_type() == Variant::Type::NIL) {
			return {memnew(NDRange(0, to_range_part(start_or_stop), xt::placeholders::xtuph{}))};
		}
		else if (step.get_type() == Variant::Type::NIL) {
			return {memnew(NDRange(to_range_part(start_or_stop), to_range_part(stop), xt::placeholders::xtuph{}))};
		}
		else {
			return {memnew(NDRange(to_range_part(start_or_stop), to_range_part(stop), to_range_part(step)))};
		}
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG(nullptr, error.what());
	}
}

uint64_t nd::size_of_dtype_in_bytes(DType dtype) {
	return va::size_of_dtype_in_bytes(dtype);
}

Ref<NDArray> nd::as_array(Variant array, nd::DType dtype) {
	auto type = array.get_type();

	// Can we take a view?
	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>(static_cast<Object *>(array))) {
			if (dtype == nd::DType::DTypeMax || ndarray->dtype() == dtype) {
				return array;
			}
		}
	}

	// Ok, we need a copy of the data.
	return nd::array(array, dtype);
}

Ref<NDArray> nd::array(Variant array, nd::DType dtype) {
	try {
		va::VArray existing_array = variant_as_array(array);

		// Default value.
		if (dtype == nd::DType::DTypeMax) {
			dtype = existing_array.dtype();
		}

		auto result = va::copy_as_dtype(existing_array, dtype);
		return {memnew(NDArray(result))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::empty(Variant shape, nd::DType dtype) {
	try {
		const auto shape_array = variant_as_shape(shape);

		return {memnew(NDArray(va::empty(dtype, shape_array)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::full(const Variant& shape, const Variant& fill_value, nd::DType dtype) {
	try {
		const auto shape_array = variant_as_shape(shape);

		switch (fill_value.get_type()) {
			case Variant::BOOL: {
				if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Bool;
				const auto value = va::constant_to_dtype(static_cast<bool>(fill_value), dtype);
				return {memnew(NDArray(va::full(value, shape_array)))};
			}
			case Variant::INT: {
				if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Int64;
				const auto value = va::constant_to_dtype(static_cast<int64_t>(fill_value), dtype);
				return {memnew(NDArray(va::full(value, shape_array)))};
			}
			case Variant::FLOAT: {
				if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Float64;
				const auto value = va::constant_to_dtype(static_cast<double_t>(fill_value), dtype);
				return {memnew(NDArray(va::full(value, shape_array)))};
			}
			default: {
				va::VArray result = va::empty(dtype, shape_array);
				result.set_with_array(variant_as_array(fill_value));
				return {memnew(NDArray(result))};
			}
		}

		ERR_FAIL_V_MSG({}, "The fill value must be a number literal (for now).");
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::zeros(Variant shape, nd::DType dtype) {
	return full(shape, 0, dtype);
}

Ref<NDArray> nd::ones(Variant shape, nd::DType dtype) {
	return full(shape, 1, dtype);
}

Ref<NDArray> nd::linspace(Variant start, Variant stop, int64_t num, bool endpoint, DType dtype) {
#ifdef NUMDOT_DISABLE_ALLOCATION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ALLOCATION_FUNCTIONS to enable it.");
#else
	if (dtype == DType::DTypeMax) {
		dtype = start.get_type() == Variant::FLOAT || stop.get_type() == Variant::FLOAT
			? nd::DType::Float64
			: nd::DType::Float32;
	}

	try {
		auto result = std::visit([start, stop, num, endpoint](auto t) {
			using T = std::decay_t<decltype(t)>;

			if constexpr (std::is_floating_point_v<T>) {
				auto store = std::make_shared<xt::xarray<T>>(xt::linspace(static_cast<double_t>(start), static_cast<double_t>(stop), num, endpoint));
				return va::from_store(store);
			}
			else {
				auto store = std::make_shared<xt::xarray<T>>(xt::linspace(static_cast<int64_t>(start), static_cast<int64_t>(stop), num, endpoint));
				return va::from_store(store);
			}
		}, va::dtype_to_variant(dtype));
		return {memnew(NDArray(result))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
#endif
}

Ref<NDArray> nd::arange(Variant start_or_stop, Variant stop, Variant step, DType dtype) {
#ifdef NUMDOT_DISABLE_ALLOCATION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ALLOCATION_FUNCTIONS to enable it.");
#else
	if (dtype == DType::DTypeMax) {
		dtype = start_or_stop.get_type() == Variant::FLOAT || stop.get_type() == Variant::FLOAT || step.get_type() == Variant::FLOAT
			? nd::DType::Float64
			: nd::DType::Int64;
	}

	// Support arange(x) syntax
	if (stop.get_type() == Variant::NIL) {
		stop = start_or_stop;
		start_or_stop = 0;
	}

	try {
		auto result = std::visit([start_or_stop, stop, step](auto t) {
			using T = std::decay_t<decltype(t)>;

			if constexpr (std::is_floating_point_v<T>) {
				auto store = std::make_shared<xt::xarray<T>>(xt::arange(static_cast<double_t>(start_or_stop), static_cast<double_t>(stop), static_cast<double_t>(step)));
				return va::from_store(store);
			}
			else {
				auto store = std::make_shared<xt::xarray<T>>(xt::arange(static_cast<int64_t>(start_or_stop), static_cast<int64_t>(stop), static_cast<int64_t>(step)));
				return va::from_store(store);
			}
		}, va::dtype_to_variant(dtype));
		return {memnew(NDArray(result))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
#endif
}

Ref<NDArray> nd::transpose(Variant a, Variant permutation) {
	try {
		va::VArray a_ = variant_as_array(a);
		// TODO It's not exactly a shape, but 'int array' is close enough.
		//  We should probably decouple them when we add better shape checks.
		const auto permutation_ = variant_as_strides(permutation);

		return {memnew(NDArray(va::transpose(a_, permutation_)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::reshape(Variant a, Variant shape) {
	try {
		va::VArray a_ = variant_as_array(a);
		// TODO It's not exactly a shape, but 'int array' is close enough.
		//  We should probably decouple them when we add better shape checks.
		const auto new_shape_ = variant_as_strides(shape);

		return {memnew(NDArray(va::reshape(a_, new_shape_)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::swapaxes(Variant v, int64_t a, int64_t b) {
	return map_variants_as_arrays([a, b](const va::VArray& v) { return va::swapaxes(v, a, b); }, v);
}

Ref<NDArray> nd::moveaxis(Variant v, int64_t src, int64_t dst) {
	return map_variants_as_arrays([src, dst](const va::VArray& v) { return va::moveaxis(v, src, dst); }, v);
}

Ref<NDArray> nd::flip(Variant v, int64_t axis) {
	return map_variants_as_arrays([axis](const va::VArray& v) { return va::flip(v, axis); }, v);
}

Ref<NDArray> nd::stack(Variant v, int64_t axis) {
	return map_variants_as_arrays([axis](const va::VArray& v) {
		return va::moveaxis(v, 0, axis);
	}, v);
}

Ref<NDArray> nd::unstack(Variant v, int64_t axis) {
	return map_variants_as_arrays([axis](const va::VArray& v) {
		return va::moveaxis(v, axis, 0);
	}, v);
}

Ref<NDArray> nd::add(Variant a, Variant b) {
	return BINARY_MAP(add, a, b);
}

Ref<NDArray> nd::subtract(Variant a, Variant b) {
	return BINARY_MAP(subtract, a, b);
}

Ref<NDArray> nd::multiply(Variant a, Variant b) {
	return BINARY_MAP(multiply, a, b);
}

Ref<NDArray> nd::divide(Variant a, Variant b) {
	return BINARY_MAP(divide, a, b);
}

Ref<NDArray> nd::remainder(Variant a, Variant b) {
	return BINARY_MAP(remainder, a, b);
}

Ref<NDArray> nd::pow(Variant a, Variant b) {
	return BINARY_MAP(pow, a, b);
}

Ref<NDArray> nd::minimum(Variant a, Variant b) {
	return BINARY_MAP(minimum, a, b);
}

Ref<NDArray> nd::maximum(Variant a, Variant b) {
	return BINARY_MAP(maximum, a, b);
}

Ref<NDArray> nd::clip(Variant a, Variant min, Variant max) {
	return TERNARY_MAP(clip, a, min, max);
}

Ref<NDArray> nd::sign(Variant a) {
	return UNARY_MAP(sign, a);
}

Ref<NDArray> nd::abs(Variant a) {
	return UNARY_MAP(abs, a);
}

Ref<NDArray> nd::square(Variant a) {
	return UNARY_MAP(square, a);
}

Ref<NDArray> nd::sqrt(Variant a) {
	return UNARY_MAP(sqrt, a);
}

Ref<NDArray> nd::exp(Variant a) {
	return UNARY_MAP(exp, a);
}

Ref<NDArray> nd::log(Variant a) {
	return UNARY_MAP(log, a);
}

Ref<NDArray> nd::rad2deg(Variant a) {
	return UNARY_MAP(rad2deg, a);
}

Ref<NDArray> nd::deg2rad(Variant a) {
	return UNARY_MAP(deg2rad, a);
}

Ref<NDArray> nd::sin(Variant a) {
	return UNARY_MAP(sin, a);
}

Ref<NDArray> nd::cos(Variant a) {
	return UNARY_MAP(cos, a);
}

Ref<NDArray> nd::tan(Variant a) {
	return UNARY_MAP(tan, a);
}

Ref<NDArray> nd::asin(Variant a) {
	return UNARY_MAP(asin, a);
}

Ref<NDArray> nd::acos(Variant a) {
	return UNARY_MAP(acos, a);
}

Ref<NDArray> nd::atan(Variant a) {
	return UNARY_MAP(atan, a);
}

Ref<NDArray> nd::atan2(Variant x1, Variant x2) {
	return BINARY_MAP(atan2, x1, x2);
}

Ref<NDArray> nd::sinh(Variant a) {
	return UNARY_MAP(sinh, a);
}

Ref<NDArray> nd::cosh(Variant a) {
	return UNARY_MAP(cosh, a);
}

Ref<NDArray> nd::tanh(Variant a) {
	return UNARY_MAP(tanh, a);
}

Ref<NDArray> nd::asinh(Variant a) {
	return UNARY_MAP(asinh, a);
}

Ref<NDArray> nd::acosh(Variant a) {
	return UNARY_MAP(acosh, a);
}

Ref<NDArray> nd::atanh(Variant a) {
	return UNARY_MAP(atanh, a);
}

Ref<NDArray> nd::sum(Variant a, Variant axes) {
	return REDUCTION1(sum, a, axes);
}

Ref<NDArray> nd::prod(Variant a, Variant axes) {
	return REDUCTION1(prod, a, axes);
}

Ref<NDArray> nd::mean(Variant a, Variant axes) {
	return REDUCTION1(mean, a, axes);
}

Ref<NDArray> nd::var(Variant a, Variant axes) {
	return REDUCTION1(var, a, axes);
}

Ref<NDArray> nd::std(Variant a, Variant axes) {
	return REDUCTION1(std, a, axes);
}

Ref<NDArray> nd::max(Variant a, Variant axes) {
	return REDUCTION1(max, a, axes);
}

Ref<NDArray> nd::min(Variant a, Variant axes) {
	return REDUCTION1(min, a, axes);
}

Ref<NDArray> nd::norm(Variant a, Variant ord, Variant axes) {
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

Ref<NDArray> nd::floor(Variant a) {
	return UNARY_MAP(floor, a);
}

Ref<NDArray> nd::ceil(Variant a) {
	return UNARY_MAP(ceil, a);
}

Ref<NDArray> nd::round(Variant a) {
	return UNARY_MAP(round, a);
}

Ref<NDArray> nd::trunc(Variant a) {
	return UNARY_MAP(trunc, a);
}

Ref<NDArray> nd::rint(Variant a) {
	// Actually uses nearbyint because rint can throw, which is undesirable in our case, and unlike numpy's behavior.
	return UNARY_MAP(nearbyint, a);
}

Ref<NDArray> nd::equal(Variant a, Variant b) {
	return BINARY_MAP(equal_to, a, b);
}

Ref<NDArray> nd::not_equal(Variant a, Variant b) {
	return BINARY_MAP(not_equal_to, a, b);
}

Ref<NDArray> nd::greater(Variant a, Variant b) {
	return BINARY_MAP(greater, a, b);
}

Ref<NDArray> nd::greater_equal(Variant a, Variant b) {
	return BINARY_MAP(greater_equal, a, b);
}

Ref<NDArray> nd::less(Variant a, Variant b) {
	return BINARY_MAP(less, a, b);
}

Ref<NDArray> nd::less_equal(Variant a, Variant b) {
	return BINARY_MAP(less_equal, a, b);
}

Ref<NDArray> nd::logical_and(Variant a, Variant b) {
	return BINARY_MAP(logical_and, a, b);
}

Ref<NDArray> nd::logical_or(Variant a, Variant b) {
	return BINARY_MAP(logical_or, a, b);
}

Ref<NDArray> nd::logical_xor(Variant a, Variant b) {
    return BINARY_MAP(logical_xor, a, b);
}

Ref<NDArray> nd::logical_not(Variant a) {
	return UNARY_MAP(logical_not, a);
}

Ref<NDArray> nd::all(Variant a, Variant axes) {
    return REDUCTION1(all, a, axes);
}

Ref<NDArray> nd::any(Variant a, Variant axes) {
    return REDUCTION1(any, a, axes);
}

Ref<NDArray> nd::dot(Variant a, Variant b) {
    return BINARY_MAP(dot, a, b);
}

Ref<NDArray> nd::reduce_dot(Variant a, Variant b, Variant axes) {
	return REDUCTION2(reduce_dot, a, b, axes);
}

Ref<NDArray> nd::matmul(Variant a, Variant b) {
	return BINARY_MAP(matmul, a, b);
}
