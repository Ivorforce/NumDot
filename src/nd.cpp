#include "nd.h"

#include <cmath>                                            // for double_t
#include <cstddef>                                          // for size_t
#include <stdexcept>                                        // for runtime_e...
#include <type_traits>                                      // for decay_t
#include <variant>                                          // for visit
#include <vector>                                           // for vector
#include "conversion_array.h"                               // for variant_a...
#include "conversion_axes.h"                                // for variant_t...
#include "conversion_range.h"                               // for to_range_...
#include "conversion_shape.h"                               // for variant_a...
#include "godot_cpp/classes/ref.hpp"                        // for Ref
#include "godot_cpp/core/error_macros.hpp"                  // for ERR_FAIL_...
#include "godot_cpp/core/memory.hpp"                        // for _post_ini...
#include "ndarray.h"                                        // for NDArray
#include "ndrange.h"                                        // for NDRange
#include "varray.h"                                         // for DType
#include "vcompute.h"                                       // for function_...
#include "xtensor/xbuilder.hpp"                             // for arange
#include "xtensor/xiterator.hpp"                            // for operator==
#include "xtensor/xlayout.hpp"                              // for layout_type
#include "xtensor/xmath.hpp"                                // for pow_fun
#include "xtensor/xoperation.hpp"                           // for divides
#include "xtensor/xslice.hpp"                               // for xtuph
#include "xtensor/xtensor_forward.hpp"                      // for xarray


using namespace godot;

void nd::_bind_methods() {
	// For the macros, we need to have the values in our namespace.
	using namespace va;
	BIND_ENUM_CONSTANT(Float64);
	BIND_ENUM_CONSTANT(Float32);
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

	godot::ClassDB::bind_static_method("nd", D_METHOD("sign", "a"), &nd::sign);
	godot::ClassDB::bind_static_method("nd", D_METHOD("abs", "a"), &nd::abs);
	godot::ClassDB::bind_static_method("nd", D_METHOD("sqrt", "a"), &nd::sqrt);

	godot::ClassDB::bind_static_method("nd", D_METHOD("exp", "a"), &nd::exp);
	godot::ClassDB::bind_static_method("nd", D_METHOD("log", "a"), &nd::log);

	godot::ClassDB::bind_static_method("nd", D_METHOD("sin", "a"), &nd::sin);
	godot::ClassDB::bind_static_method("nd", D_METHOD("cos", "a"), &nd::cos);
	godot::ClassDB::bind_static_method("nd", D_METHOD("tan", "a"), &nd::tan);

	godot::ClassDB::bind_static_method("nd", D_METHOD("sum", "a", "axes"), &nd::sum, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("prod", "a", "axes"), &nd::sum, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("mean", "a", "axes"), &nd::mean, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("var", "a", "axes"), &nd::std, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("std", "a", "axes"), &nd::std, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("max", "a", "axes"), &nd::std, DEFVAL(nullptr), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("min", "a", "axes"), &nd::std, DEFVAL(nullptr), DEFVAL(nullptr));
}

nd::nd() {
}

nd::~nd() {
	// Add your cleanup here.
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
			dtype = va::dtype(existing_array);
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

template <typename V>
Ref<NDArray> _full(Variant shape, V value, nd::DType dtype) {
	try {
		std::vector<size_t> shape_array = variant_as_shape<size_t, std::vector<size_t>>(shape);

		return {memnew(NDArray(va::full(dtype, value, shape_array)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::full(Variant shape, Variant fill_value, nd::DType dtype) {
	switch (fill_value.get_type()) {
		case Variant::INT:
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Int64;
			return _full(shape, int64_t(fill_value), dtype);
		case Variant::FLOAT:
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Float64;
			return _full(shape, double_t(fill_value), dtype);
		default:
			ERR_FAIL_V_MSG({}, "The fill value must be a number literal (for now).");
	}
}

Ref<NDArray> nd::zeros(Variant shape, nd::DType dtype) {
	return _full(shape, 0, dtype);
}

Ref<NDArray> nd::ones(Variant shape, nd::DType dtype) {
	return _full(shape, 1, dtype);
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
	try {
		va::VArray v_ = variant_as_array(v);
		return {memnew(NDArray(va::swapaxes(v_, a, b)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::moveaxis(Variant v, int64_t src, int64_t dst) {
	try {
		va::VArray v_ = variant_as_array(v);
		return {memnew(NDArray(va::moveaxis(v_, src, dst)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::flip(Variant v, int64_t axis) {
	try {
		va::VArray v_ = variant_as_array(v);
		return {memnew(NDArray(va::flip(v_, axis)))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

// The first parameter is the one used by the xarray operation, while the second is used for type deduction.
// It's ok if they're the same.
template <typename FX, typename PromotionRule>
inline Ref<NDArray> binary_operation(Variant a, Variant b) {
	try {
		va::VArray a_ = variant_as_array(a);
		va::VArray b_ = variant_as_array(b);

		auto comp_a = va::to_compute_variant(a_);
		auto comp_b = va::to_compute_variant(b_);

		auto result = va::xoperation<PromotionRule>(va::XFunction<FX> {}, comp_a, comp_b);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::add(Variant a, Variant b) {
	// godot::UtilityFunctions::print(value);
	return binary_operation<xt::detail::plus, va::promote::function_result<xt::detail::plus>>(a, b);
}

Ref<NDArray> nd::subtract(Variant a, Variant b) {
	return binary_operation<xt::detail::minus, va::promote::function_result<xt::detail::minus>>(a, b);
}

Ref<NDArray> nd::multiply(Variant a, Variant b) {
	return binary_operation<xt::detail::multiplies, va::promote::function_result<xt::detail::multiplies>>(a, b);
}

Ref<NDArray> nd::divide(Variant a, Variant b) {
	return binary_operation<xt::detail::divides, va::promote::function_result<xt::detail::divides>>(a, b);
}

Ref<NDArray> nd::remainder(Variant a, Variant b) {
	return binary_operation<xt::math::remainder_fun, va::promote::function_result<xt::math::remainder_fun>>(a, b);
}

Ref<NDArray> nd::pow(Variant a, Variant b) {
	return binary_operation<xt::math::pow_fun, va::promote::function_result<xt::math::pow_fun>>(a, b);
}


template <typename FX, typename PromotionRule>
inline Ref<NDArray> unary_operation(Variant a) {
	try {
		auto a_ = variant_as_array(a);

		auto comp_a = va::to_compute_variant(a_);

		auto result = va::xoperation<PromotionRule>(va::XFunction<FX> {}, comp_a);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::sign(Variant a) {
	return unary_operation<xt::math::sign_fun, va::promote::common_type>(a);
}

Ref<NDArray> nd::abs(Variant a) {
	return unary_operation<xt::math::abs_fun, va::promote::function_result<xt::math::abs_fun>>(a);
}

Ref<NDArray> nd::sqrt(Variant a) {
	return unary_operation<xt::math::sqrt_fun, va::promote::function_result<xt::math::sqrt_fun>>(a);
}

Ref<NDArray> nd::exp(Variant a) {
	return unary_operation<xt::math::exp_fun, va::promote::function_result<xt::math::exp_fun>>(a);
}

Ref<NDArray> nd::log(Variant a) {
	return unary_operation<xt::math::log_fun, va::promote::function_result<xt::math::log_fun>>(a);
}

Ref<NDArray> nd::sin(Variant a) {
	return unary_operation<xt::math::sin_fun, va::promote::function_result<xt::math::sin_fun>>(a);
}

Ref<NDArray> nd::cos(Variant a) {
	return unary_operation<xt::math::cos_fun, va::promote::function_result<xt::math::cos_fun>>(a);
}

Ref<NDArray> nd::tan(Variant a) {
	return unary_operation<xt::math::tan_fun, va::promote::function_result<xt::math::tan_fun>>(a);
}

template <typename FX, typename PromotionRule>
inline Ref<NDArray> reduction(Variant a, Variant axes) {
	try {
		auto axes_ = variant_to_axes(axes);
		auto a_ = variant_as_array(a);

		auto result = va::xreduction<PromotionRule>(
			FX{}, axes_, va::to_compute_variant(a_)
		);

		return {memnew(NDArray(result))};
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

#define Reducer(Name, fun_name)\
	Name() = default;\
	Name(const Name&) = default;\
	Name(Name&&) noexcept = default;\
	Name& operator=(const Name&) = default;\
	Name& operator=(Name&&) noexcept = default;\
	~Name() = default;\
\
	template <typename GivenAxes, typename A>\
	auto operator()(GivenAxes&& axes, A&& a) const {\
		return xt::fun_name(std::forward<A>(a), std::forward<GivenAxes>(axes));\
	}\
\
	template <typename A>\
	auto operator()(A&& a) const {\
		return xt::fun_name(std::forward<A>(a));\
	}

struct Sum { Reducer(Sum, sum) };

Ref<NDArray> nd::sum(Variant a, Variant axes) {
	return reduction<Sum, va::promote::common_type>(a, axes);
}

struct Prod { Reducer(Prod, prod) };

Ref<NDArray> nd::prod(Variant a, Variant axes) {
	return reduction<Prod, va::promote::at_least_int32>(a, axes);
}

struct Mean { Reducer(Mean, mean) };

Ref<NDArray> nd::mean(Variant a, Variant axes) {
	return reduction<Mean, va::promote::matching_float_or_default<double_t>>(a, axes);
}

struct Variance { Reducer(Variance, variance) };

Ref<NDArray> nd::var(Variant a, Variant axes) {
	return reduction<Variance, va::promote::matching_float_or_default<double_t>>(a, axes);
}

struct Std { Reducer(Std, stddev) };

Ref<NDArray> nd::std(Variant a, Variant axes) {
	return reduction<Std, va::promote::matching_float_or_default<double_t>>(a, axes);
}

struct Amax { Reducer(Amax, amax) };

Ref<NDArray> nd::max(Variant a, Variant axes) {
	return reduction<Amax, va::promote::common_type>(a, axes);
}

struct Amin { Reducer(Amin, amin) };

Ref<NDArray> nd::min(Variant a, Variant axes) {
	return reduction<Amin, va::promote::common_type>(a, axes);
}
