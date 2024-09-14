#include "nd.h"

#include <vatensor/vmath.h>                  // for abs, add, divide, exp, log
#include <vatensor/reduce.h>                // for max, mean, min, prod, std
#include <vatensor/trigonometry.h>          // for cos, sin, tan
#include <vatensor/round.h>					// for rounding functions
#include <cmath>                            // for double_t
#include <cstddef>                          // for size_t, ptrdiff_t
#include <functional>                       // for function
#include <memory>                           // for make_shared
#include <stdexcept>                        // for runtime_error
#include <type_traits>                      // for decay_t
#include <utility>                          // for move
#include <variant>                          // for visit
#include <vector>                           // for vector
#include <vatensor/comparison.h>
#include <vatensor/logical.h>
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
#include "vatensor/rearrange.h"             // for flip, moveaxis, reshape
#include "vatensor/varray.h"                // for VArray, DType, Axes, cons...
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

	godot::ClassDB::bind_static_method("nd", D_METHOD("add", "a", "b"), &nd::add);
	godot::ClassDB::bind_static_method("nd", D_METHOD("subtract", "a", "b"), &nd::subtract);
	godot::ClassDB::bind_static_method("nd", D_METHOD("multiply", "a", "b"), &nd::multiply);
	godot::ClassDB::bind_static_method("nd", D_METHOD("divide", "a", "b"), &nd::divide);
	godot::ClassDB::bind_static_method("nd", D_METHOD("remainder", "a", "b"), &nd::remainder);
	godot::ClassDB::bind_static_method("nd", D_METHOD("pow", "a", "b"), &nd::pow);

	godot::ClassDB::bind_static_method("nd", D_METHOD("minimum", "a", "b"), &nd::minimum);
	godot::ClassDB::bind_static_method("nd", D_METHOD("maximum", "a", "b"), &nd::maximum);

	godot::ClassDB::bind_static_method("nd", D_METHOD("sign", "a"), &nd::sign);
	godot::ClassDB::bind_static_method("nd", D_METHOD("abs", "a"), &nd::abs);
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

	godot::ClassDB::bind_static_method("nd", D_METHOD("sinh", "a"), &nd::sinh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("cosh", "a"), &nd::cosh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("tanh", "a"), &nd::tanh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("asinh", "a"), &nd::asinh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("acosh", "a"), &nd::acosh);
	godot::ClassDB::bind_static_method("nd", D_METHOD("atanh", "a"), &nd::atanh);

	godot::ClassDB::bind_static_method("nd", D_METHOD("sum", "a", "axes"), &nd::sum, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("prod", "a", "axes"), &nd::sum, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("mean", "a", "axes"), &nd::mean, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("var", "a", "axes"), &nd::var, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("std", "a", "axes"), &nd::std, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("max", "a", "axes"), &nd::max, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("min", "a", "axes"), &nd::min, DEFVAL(nullptr), DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("floor", "a"), &nd::floor);
    godot::ClassDB::bind_static_method("nd", D_METHOD("ceil", "a"), &nd::ceil);
    godot::ClassDB::bind_static_method("nd", D_METHOD("round", "a"), &nd::round);
    godot::ClassDB::bind_static_method("nd", D_METHOD("trunc", "a"), &nd::trunc);
	godot::ClassDB::bind_static_method("nd", D_METHOD("rint", "a"), &nd::rint);

	godot::ClassDB::bind_static_method("nd", D_METHOD("equal", "a", "b"), &nd::equal);

	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_and", "a", "b"), &nd::logical_and);
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_or", "a", "b"), &nd::logical_or);
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_not", "a"), &nd::logical_not);
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

StringName nd::newaxis() {
	return ::newaxis();
}

StringName nd::ellipsis() {
	return ::ellipsis();
}

Ref<NDRange> nd::from(int64_t start) {
	return {memnew(NDRange(start, xt::placeholders::xtuph{}, xt::placeholders::xtuph{}))};
}

Ref<NDRange> nd::to(int64_t stop) {
	return {memnew(NDRange(xt::placeholders::xtuph{}, stop, xt::placeholders::xtuph{}))};
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
		std::vector<size_t> shape_array = variant_as_shape<size_t, std::vector<size_t>>(shape);

		return {memnew(NDArray(va::empty(dtype, shape_array)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> _full(Variant shape, va::VConstant value) {
	try {
		std::vector<size_t> shape_array = variant_as_shape<size_t, std::vector<size_t>>(shape);

		return {memnew(NDArray(va::full(value, shape_array)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::full(const Variant& shape, const Variant& fill_value, nd::DType dtype) {
	switch (fill_value.get_type()) {
		case Variant::INT:
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Int64;
			return _full(shape, va::constant_to_dtype(static_cast<int64_t>(fill_value), dtype));
		case Variant::FLOAT:
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Float64;
			return _full(shape, va::constant_to_dtype(static_cast<double_t>(fill_value), dtype));
		default:
			ERR_FAIL_V_MSG({}, "The fill value must be a number literal (for now).");
	}
}

Ref<NDArray> nd::zeros(Variant shape, nd::DType dtype) {
	return _full(std::move(shape), va::constant_to_dtype(0, dtype));
}

Ref<NDArray> nd::ones(Variant shape, nd::DType dtype) {
	return _full(std::move(shape), va::constant_to_dtype(1, dtype));
}

Ref<NDArray> nd::linspace(Variant start, Variant stop, int64_t num, bool endpoint, DType dtype) {
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
}

Ref<NDArray> nd::arange(Variant start_or_stop, Variant stop, Variant step, DType dtype) {
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
}

Ref<NDArray> nd::transpose(Variant a, Variant permutation) {
	try {
		va::VArray a_ = variant_as_array(a);
		// TODO It's not exactly a shape, but 'int array' is close enough.
		//  We should probably decouple them when we add better shape checks.
		auto permutation_ = variant_as_shape<std::ptrdiff_t, va::strides_type>(permutation);

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
		auto new_shape_ = variant_as_shape<std::ptrdiff_t, va::strides_type>(shape);

		return {memnew(NDArray(va::reshape(a_, new_shape_)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::swapaxes(Variant v, int64_t a, int64_t b) {
	return map_variants_as_arrays([a, b](const va::VArray &v) { return va::swapaxes(v, a, b); }, v);
}

Ref<NDArray> nd::moveaxis(Variant v, int64_t src, int64_t dst) {
	return map_variants_as_arrays([src, dst](const va::VArray &v) { return va::moveaxis(v, src, dst); }, v);
}

Ref<NDArray> nd::flip(Variant v, int64_t axis) {
	return map_variants_as_arrays([axis](const va::VArray &v) { return va::flip(v, axis); }, v);
}

Ref<NDArray> nd::add(Variant a, Variant b) {
	// godot::UtilityFunctions::print(value);
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::add(a, b); }, a, b);
}

Ref<NDArray> nd::subtract(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::subtract(a, b); }, a, b);
}

Ref<NDArray> nd::multiply(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::multiply(a, b); }, a, b);
}

Ref<NDArray> nd::divide(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::divide(a, b); }, a, b);
}

Ref<NDArray> nd::remainder(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::remainder(a, b); }, a, b);
}

Ref<NDArray> nd::pow(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::pow(a, b); }, a, b);
}

Ref<NDArray> nd::minimum(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::minimum(a, b); }, a, b);
}

Ref<NDArray> nd::maximum(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::maximum(a, b); }, a, b);
}

Ref<NDArray> nd::sign(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::sign(varray); }, a);
}

Ref<NDArray> nd::abs(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::abs(varray); }, a);
}

Ref<NDArray> nd::sqrt(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::sqrt(varray); }, a);
}

Ref<NDArray> nd::exp(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::exp(varray); }, a);
}

Ref<NDArray> nd::log(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::log(varray); }, a);
}

Ref<NDArray> nd::rad2deg(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::rad2deg(varray); }, a);
}

Ref<NDArray> nd::deg2rad(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::deg2rad(varray); }, a);
}

Ref<NDArray> nd::sin(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::sin(varray); }, a);
}

Ref<NDArray> nd::cos(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::cos(varray); }, a);
}

Ref<NDArray> nd::tan(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::tan(varray); }, a);
}

Ref<NDArray> nd::asin(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::asin(varray); }, a);
}

Ref<NDArray> nd::acos(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::acos(varray); }, a);
}

Ref<NDArray> nd::atan(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::atan(varray); }, a);
}

Ref<NDArray> nd::sinh(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::sinh(varray); }, a);
}

Ref<NDArray> nd::cosh(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::cosh(varray); }, a);
}

Ref<NDArray> nd::tanh(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::tanh(varray); }, a);
}

Ref<NDArray> nd::asinh(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::asinh(varray); }, a);
}

Ref<NDArray> nd::acosh(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::acosh(varray); }, a);
}

Ref<NDArray> nd::atanh(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray){ return va::atanh(varray); }, a);
}

inline Ref<NDArray> reduction(std::function<va::VArray(const va::VArray&, const va::Axes&)> visitor, Variant a, Variant axes) {
	try {
		const auto axes_ = variant_to_axes(axes);
		const auto a_ = variant_as_array(a);

		const auto result = visitor(a_, axes_);

		return {memnew(NDArray(result))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::sum(Variant a, Variant axes) {
	return reduction([](const va::VArray& array, const va::Axes& axes) { return va::sum(array, axes); }, a, axes);
}

Ref<NDArray> nd::prod(Variant a, Variant axes) {
	return reduction([](const va::VArray& array, const va::Axes& axes) { return va::prod(array, axes); }, a, axes);
}

Ref<NDArray> nd::mean(Variant a, Variant axes) {
	return reduction([](const va::VArray& array, const va::Axes& axes) { return va::mean(array, axes); }, a, axes);
}

Ref<NDArray> nd::var(Variant a, Variant axes) {
	return reduction([](const va::VArray& array, const va::Axes& axes) { return va::var(array, axes); }, a, axes);
}

Ref<NDArray> nd::std(Variant a, Variant axes) {
	return reduction([](const va::VArray& array, const va::Axes& axes) { return va::std(array, axes); }, a, axes);
}

Ref<NDArray> nd::max(Variant a, Variant axes) {
	return reduction([](const va::VArray& array, const va::Axes& axes) { return va::max(array, axes); }, a, axes);
}

Ref<NDArray> nd::min(Variant a, Variant axes) {
	return reduction([](const va::VArray& array, const va::Axes& axes) { return va::min(array, axes); }, a, axes);
}

Ref<NDArray> nd::floor(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray) { return va::floor(varray); }, a);
}

Ref<NDArray> nd::ceil(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray) { return va::ceil(varray); }, a);
}

Ref<NDArray> nd::round(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray) { return va::round(varray); }, a);
}

Ref<NDArray> nd::trunc(Variant a) {
	return map_variants_as_arrays([](const va::VArray &varray) { return va::trunc(varray); }, a);
}

Ref<NDArray> nd::rint(Variant a) {
	// Actually uses nearbyint because rint can throw, which is undesirable in our case, and unlike numpy's behavior.
	return map_variants_as_arrays([](const va::VArray &varray) { return va::nearbyint(varray); }, a);
}

Ref<NDArray> nd::equal(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::equal_to(a, b); }, a, b);
}

Ref<NDArray> nd::logical_and(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::logical_and(a, b); }, a, b);
}

Ref<NDArray> nd::logical_or(Variant a, Variant b) {
	return map_variants_as_arrays([](const va::VArray &a, const va::VArray &b) { return va::logical_or(a, b); }, a, b);
}

Ref<NDArray> nd::logical_not(Variant a) {
	return map_variants_as_arrays([](const va::VArray &a) { return va::logical_not(a); }, a);
}
