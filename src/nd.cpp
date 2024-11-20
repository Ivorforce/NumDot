#include "nd.hpp"

#include <vatensor/comparison.hpp>            // for equal_to, greater, greate...
#include <vatensor/linalg.hpp>                // for reduce_dot, dot, matmul
#include <vatensor/logical.hpp>               // for logical_and, logical_not
#include <vatensor/bitwise.hpp>               // for bitwise_and, bitwise_not
#include <vatensor/reduce.hpp>                // for all, any, max, mean, median
#include <vatensor/round.hpp>                 // for ceil, floor, nearbyint
#include <vatensor/trigonometry.hpp>          // for acos, acosh, asin, asinh
#include <vatensor/vassign.hpp>               // for assign
#include <vatensor/vmath.hpp>                 // for abs, add, clip, deg2rad
#include <cmath>                            // for double_t, isinf
#include <optional>                         // for optional
#include <stdexcept>                        // for runtime_error
#include <memory>                           // shared_ptr
#include <nd.hpp>
#include <type_traits>                      // for decay_t
#include <utility>                          // for forward
#include <variant>                          // for visit
#include <gdconvert/conversion_scalar.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <vatensor/stride_tricks.hpp>
#include <vatensor/vcarray.hpp>
#include <vatensor/vsignal.hpp>
#include <vatensor/vio.hpp>
#include <vatensor/dtype.hpp>
#include <vatensor/xscalar_store.hpp>
#include <vatensor/xtensor_store.hpp>
#include "gdconvert/conversion_array.hpp"     // for variant_as_array
#include "gdconvert/conversion_ints.hpp"      // for variant_to_axes, variant_...
#include "gdconvert/conversion_slice.hpp"     // for ellipsis, newaxis
#include "godot_cpp/classes/ref.hpp"        // for Ref
#include "godot_cpp/core/error_macros.hpp"  // for ERR_FAIL_V_MSG, ERR_FAIL_...
#include "godot_cpp/core/memory.hpp"        // for _post_initialize, memnew
#include "ndarray.hpp"                        // for NDArray
#include "ndutil.hpp"
#include "vatensor/create.hpp"              // for full, empty
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
	BIND_ENUM_CONSTANT(Complex64);
	BIND_ENUM_CONSTANT(Complex128);
	BIND_ENUM_CONSTANT(Int8);
	BIND_ENUM_CONSTANT(Int16);
	BIND_ENUM_CONSTANT(Int32);
	BIND_ENUM_CONSTANT(Int64);
	BIND_ENUM_CONSTANT(UInt8);
	BIND_ENUM_CONSTANT(UInt16);
	BIND_ENUM_CONSTANT(UInt32);
	BIND_ENUM_CONSTANT(UInt64);
	BIND_ENUM_CONSTANT(DTypeMax);

	BIND_ENUM_CONSTANT(Constant);
	BIND_ENUM_CONSTANT(Symmetric);
	BIND_ENUM_CONSTANT(Reflect);
	BIND_ENUM_CONSTANT(Wrap);
	BIND_ENUM_CONSTANT(Edge);

	godot::ClassDB::bind_static_method("nd", D_METHOD("newaxis"), &nd::newaxis);
	godot::ClassDB::bind_static_method("nd", D_METHOD("ellipsis"), &nd::ellipsis);

	godot::ClassDB::bind_static_method("nd", D_METHOD("from", "start"), &nd::from);
	godot::ClassDB::bind_static_method("nd", D_METHOD("to", "stop"), &nd::to);
	godot::ClassDB::bind_static_method("nd", D_METHOD("range", "start_or_stop", "stop", "step"), &nd::range, DEFVAL(no_value()), DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("size_of_dtype_in_bytes", "dtype"), &nd::size_of_dtype_in_bytes);

	godot::ClassDB::bind_static_method("nd", D_METHOD("as_array", "array", "dtype"), &nd::as_array, DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("array", "array", "dtype"), &nd::array, DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("copy", "array"), &nd::copy);

	godot::ClassDB::bind_static_method("nd", D_METHOD("bool_", "array"), &nd::bool_);
	godot::ClassDB::bind_static_method("nd", D_METHOD("float32", "array"), &nd::float32);
	godot::ClassDB::bind_static_method("nd", D_METHOD("float64", "array"), &nd::float64);
	godot::ClassDB::bind_static_method("nd", D_METHOD("complex64", "array"), &nd::complex64);
	godot::ClassDB::bind_static_method("nd", D_METHOD("complex128", "array"), &nd::complex128);
	godot::ClassDB::bind_static_method("nd", D_METHOD("int8", "array"), &nd::int8);
	godot::ClassDB::bind_static_method("nd", D_METHOD("int16", "array"), &nd::int16);
	godot::ClassDB::bind_static_method("nd", D_METHOD("int32", "array"), &nd::int32);
	godot::ClassDB::bind_static_method("nd", D_METHOD("int64", "array"), &nd::int64);
	godot::ClassDB::bind_static_method("nd", D_METHOD("uint8", "array"), &nd::uint8);
	godot::ClassDB::bind_static_method("nd", D_METHOD("uint16", "array"), &nd::uint16);
	godot::ClassDB::bind_static_method("nd", D_METHOD("uint32", "array"), &nd::uint32);
	godot::ClassDB::bind_static_method("nd", D_METHOD("uint64", "array"), &nd::uint64);

	godot::ClassDB::bind_static_method("nd", D_METHOD("empty", "shape", "dtype"), &nd::empty, DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("empty_like", "model", "dtype", "shape"), &nd::empty_like, DEFVAL(nd::DType::DTypeMax), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("full", "shape", "fill_value", "dtype"), &nd::full, DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("full_like", "model", "fill_value", "dtype", "shape"), &nd::full_like, DEFVAL(nd::DType::DTypeMax), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("zeros", "shape", "dtype"), &nd::zeros, DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("ones_like", "model", "dtype", "shape"), &nd::ones_like, DEFVAL(nd::DType::DTypeMax), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("ones", "shape", "dtype"), &nd::ones, DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("zeros_like", "model", "dtype", "shape"), &nd::zeros_like, DEFVAL(nd::DType::DTypeMax), DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("eye", "shape", "k", "dtype"), &nd::eye, DEFVAL(0), DEFVAL(nd::DType::Float64));
	godot::ClassDB::bind_static_method("nd", D_METHOD("linspace", "start", "stop", "num", "endpoint", "dtype"), &nd::linspace, DEFVAL(50), DEFVAL(true), DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("arange", "start_or_stop", "stop", "step", "dtype"), &nd::arange, DEFVAL(nullptr), DEFVAL(1), DEFVAL(nd::DType::DTypeMax));

	godot::ClassDB::bind_static_method("nd", D_METHOD("transpose", "a", "permutation"), &nd::transpose, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("reshape", "a", "shape"), &nd::reshape);
	godot::ClassDB::bind_static_method("nd", D_METHOD("swapaxes", "v", "a", "b"), &nd::swapaxes);
	godot::ClassDB::bind_static_method("nd", D_METHOD("moveaxis", "v", "src", "dst"), &nd::moveaxis);
	godot::ClassDB::bind_static_method("nd", D_METHOD("flip", "v", "axis"), &nd::flip);
	godot::ClassDB::bind_static_method("nd", D_METHOD("diagonal", "v", "offset", "axis1", "axis2"), &nd::diagonal, DEFVAL(0), DEFVAL(0), DEFVAL(1));
	godot::ClassDB::bind_static_method("nd", D_METHOD("diag", "v", "offset"), &nd::diag, DEFVAL(0));
	godot::ClassDB::bind_static_method("nd", D_METHOD("trace", "v", "offset", "axis1", "axis2"), &nd::trace, DEFVAL(0), DEFVAL(0), DEFVAL(1));
	godot::ClassDB::bind_static_method("nd", D_METHOD("stack", "v", "axis"), &nd::stack, DEFVAL(0));
	godot::ClassDB::bind_static_method("nd", D_METHOD("unstack", "v", "axis"), &nd::unstack, DEFVAL(0));
	godot::ClassDB::bind_static_method("nd", D_METHOD("concatenate", "v", "axis", "dtype"), &nd::concatenate, DEFVAL(0), DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("hstack", "v", "dtype"), &nd::hstack, DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("vstack", "v", "dtype"), &nd::vstack, DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("tile", "v", "reps", "inner"), &nd::tile, DEFVAL(false));
	godot::ClassDB::bind_static_method("nd", D_METHOD("split", "v", "indices_or_section_size", "axis"), &nd::split, DEFVAL(0));
	godot::ClassDB::bind_static_method("nd", D_METHOD("hsplit", "v", "indices_or_section_size"), &nd::hsplit);
	godot::ClassDB::bind_static_method("nd", D_METHOD("vsplit", "v", "indices_or_section_size"), &nd::vsplit);

	godot::ClassDB::bind_static_method("nd", D_METHOD("real", "v"), &nd::real);
	godot::ClassDB::bind_static_method("nd", D_METHOD("imag", "v"), &nd::imag);
	godot::ClassDB::bind_static_method("nd", D_METHOD("conjugate", "v"), &nd::conjugate);
	godot::ClassDB::bind_static_method("nd", D_METHOD("angle", "v"), &nd::angle);
	godot::ClassDB::bind_static_method("nd", D_METHOD("vector_as_complex", "v", "keepdims", "dtype"), &nd::vector_as_complex, DEFVAL(false), DEFVAL(nd::DType::DTypeMax));
	godot::ClassDB::bind_static_method("nd", D_METHOD("complex_as_vector", "v"), &nd::complex_as_vector);

	godot::ClassDB::bind_static_method("nd", D_METHOD("positive", "a"), &nd::positive);
	godot::ClassDB::bind_static_method("nd", D_METHOD("negative", "a"), &nd::negative);
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

	godot::ClassDB::bind_static_method("nd", D_METHOD("sum", "a", "axes"), &nd::sum, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("prod", "a", "axes"), &nd::prod, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("mean", "a", "axes"), &nd::mean, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("median", "a", "axes"), &nd::median, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("var", "a", "axes"), &nd::variance, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("std", "a", "axes"), &nd::standard_deviation, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("max", "a", "axes"), &nd::max, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("min", "a", "axes"), &nd::min, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("norm", "a", "ord", "axes"), &nd::norm, DEFVAL(2), DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("count_nonzero", "a", "axes"), &nd::count_nonzero, DEFVAL(nullptr));

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
	godot::ClassDB::bind_static_method("nd", D_METHOD("is_close", "a", "b", "rtol", "atol", "equal_nan"), &nd::is_close, DEFVAL(1e-05), DEFVAL(1e-08), DEFVAL(false));
	godot::ClassDB::bind_static_method("nd", D_METHOD("is_nan", "a"), &nd::is_nan);
	godot::ClassDB::bind_static_method("nd", D_METHOD("is_inf", "a"), &nd::is_inf);
	godot::ClassDB::bind_static_method("nd", D_METHOD("is_finite", "a"), &nd::is_finite);

	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_and", "a", "b"), &nd::logical_and);
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_or", "a", "b"), &nd::logical_or);
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_xor", "a", "b"), &nd::logical_xor);
	godot::ClassDB::bind_static_method("nd", D_METHOD("logical_not", "a"), &nd::logical_not);
	godot::ClassDB::bind_static_method("nd", D_METHOD("all", "a", "axes"), &nd::all, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("any", "a", "axes"), &nd::any, DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("bitwise_and", "a", "b"), &nd::bitwise_and);
	godot::ClassDB::bind_static_method("nd", D_METHOD("bitwise_or", "a", "b"), &nd::bitwise_or);
	godot::ClassDB::bind_static_method("nd", D_METHOD("bitwise_xor", "a", "b"), &nd::bitwise_xor);
	godot::ClassDB::bind_static_method("nd", D_METHOD("bitwise_not", "a"), &nd::bitwise_not);
	godot::ClassDB::bind_static_method("nd", D_METHOD("bitwise_left_shift", "a", "b"), &nd::bitwise_left_shift);
	godot::ClassDB::bind_static_method("nd", D_METHOD("bitwise_right_shift", "a", "b"), &nd::bitwise_right_shift);

	godot::ClassDB::bind_static_method("nd", D_METHOD("dot", "a", "b"), &nd::dot);
	godot::ClassDB::bind_static_method("nd", D_METHOD("reduce_dot", "a", "b", "axes"), &nd::reduce_dot, DEFVAL(nullptr));
	godot::ClassDB::bind_static_method("nd", D_METHOD("matmul", "a", "b"), &nd::matmul);
	godot::ClassDB::bind_static_method("nd", D_METHOD("cross", "a", "b", "axisa", "axisb", "axisc"), &nd::cross, DEFVAL(-1), DEFVAL(-1), DEFVAL(-1));

	godot::ClassDB::bind_static_method("nd", D_METHOD("sliding_window_view", "array", "window_shape"), &nd::sliding_window_view);
	godot::ClassDB::bind_static_method("nd", D_METHOD("convolve", "array", "kernel"), &nd::convolve);

	godot::ClassDB::bind_static_method("nd", D_METHOD("default_rng", "seed"), &nd::default_rng, DEFVAL(nullptr));

	godot::ClassDB::bind_static_method("nd", D_METHOD("fft", "v", "axis"), &nd::fft, DEFVAL(-1));
	godot::ClassDB::bind_static_method("nd", D_METHOD("fft_freq", "n", "d"), &nd::fft_freq, DEFVAL(1));
	godot::ClassDB::bind_static_method("nd", D_METHOD("pad", "v", "pad_width", "pad_mode", "pad_value"), &nd::pad, DEFVAL(nd::PadMode::Constant), DEFVAL(0));

	godot::ClassDB::bind_static_method("nd", D_METHOD("load", "file_or_buffer"), &nd::load);
	godot::ClassDB::bind_static_method("nd", D_METHOD("dumpb", "array"), &nd::dumpb);
}

nd::nd() = default;
nd::~nd() = default;

template<typename Visitor, typename... Args>
Ref<NDArray> map_variants_as_arrays(Visitor&& visitor, const Args&... args) {
	try {
		const std::shared_ptr<va::VArray> result = std::forward<Visitor>(visitor)(variant_as_array(args)...);
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
		std::forward<Visitor>(visitor)(&result, variant_as_array(args)...);
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
				return { memnew(NDArray(va::store::from_scalar_variant(result))) };
			}
			else {
				return { memnew(NDArray(va::store::from_scalar(result))) };
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

		if (dtype == nd::DType::DTypeMax || shape.get_type() == Variant::NIL)
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
	map_variants_as_arrays_with_target([](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& varray) {\
        va::func(va::store::default_allocator, target, varray->data);\
    }, (varray1))

#define VARRAY_MAP2(func, varray1, varray2) \
	map_variants_as_arrays_with_target([](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b) {\
        va::func(va::store::default_allocator, target, a->data, b->data);\
    }, (varray1), (varray2))

#define VARRAY_MAP3(func, varray1, varray2, varray3) \
	map_variants_as_arrays_with_target([](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b, const std::shared_ptr<va::VArray>& c) {\
        va::func(va::store::default_allocator, target, a->data, b->data, c->data);\
    }, (varray1), (varray2), (varray3))

#define REDUCTION1(func, varray1, axes1) \
	reduction([](const va::VArrayTarget& target, const va::axes_type& axes, const va::VArray& array) {\
		va::func(va::store::default_allocator, target, array.data, axes);\
	}, [](const va::VArray& array) { return va::func(array.data); }, axes, (varray1))

#define REDUCTION2(func, varray1, varray2, axes1) \
	reduction([](const va::VArrayTarget& target, const va::axes_type& axes, const va::VArray& carray1, const va::VArray& carray2) {\
		va::func(va::store::default_allocator, target, carray1.data, carray2.data, axes);\
	}, [](const va::VArray& carray1, const va::VArray& carray2) {\
		return va::func(carray1.data, carray2.data);\
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
	if (is_no_value(stop)) {
		// Special case: nd.range(x)
		return Vector4i(0b010, 0, static_cast<int32_t>(start_or_stop), 1);
	}

	const auto type1 = start_or_stop.get_type();
	const auto type2 = stop.get_type();
	const auto type3 = step.get_type();
	ERR_FAIL_COND_V_MSG(
		(type1 != Variant::NIL && type1 != Variant::INT) || (type2 != Variant::NIL && type2 != Variant::INT) || (type3 != Variant::NIL && type3 != Variant::INT),
		Vector4i(),
		"All arguments to range must be ints or nil."
	);

	const auto mask = (0b100 * (type1 == Variant::INT)) | (0b010 * (type2 == Variant::INT)) | (0b001 * (type3 == Variant::INT));
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
Ref<NDArray> nd::complex64(const Variant& array) { return nd::as_array(array, DType::Complex64); }
Ref<NDArray> nd::complex128(const Variant& array) { return nd::as_array(array, DType::Complex128); }
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
			return { memnew(NDArray(va::empty(va::store::default_allocator, dtype, shape))) };
		}, model, dtype, shape
	);
}

Ref<NDArray> nd::empty(const Variant& shape, const nd::DType dtype) {
	try {
		const auto shape_array = variant_to_shape(shape);

		return { memnew(NDArray(va::empty(va::store::default_allocator, dtype, shape_array))) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> full(const va::shape_type& shape, nd::DType dtype, const Variant& fill_value) {
	switch (fill_value.get_type()) {
		case Variant::BOOL: {
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Bool;
			const auto value = va::static_cast_scalar(static_cast<bool>(fill_value), dtype);
			return { memnew(NDArray(va::full(va::store::default_allocator, value, shape))) };
		}
		case Variant::INT: {
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Int64;
			const auto value = va::static_cast_scalar(static_cast<int64_t>(fill_value), dtype);
			return { memnew(NDArray(va::full(va::store::default_allocator, value, shape))) };
		}
		case Variant::FLOAT: {
			if (dtype == nd::DType::DTypeMax) dtype = nd::DType::Float64;
			const auto value = va::static_cast_scalar(static_cast<double_t>(fill_value), dtype);
			return { memnew(NDArray(va::full(va::store::default_allocator, value, shape))) };
		}
		default: {
			std::shared_ptr<va::VArray> result = va::empty(va::store::default_allocator, dtype, shape);
			result->prepare_write();
			va::assign(result->data, variant_as_array(fill_value)->data);
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

		auto result = va::eye(va::store::default_allocator, dtype, used_shape, k);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::linspace(const Variant& start, const Variant& stop, const int64_t num, const bool endpoint, DType dtype) {
	try {
		const auto start_ = variant_to_vscalar(start);
		const auto stop_ = variant_to_vscalar(stop);

		auto result = va::linspace(
			va::store::default_allocator,
			start_,
			stop_,
			num,
			endpoint,
			dtype
		);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::arange(const Variant& start_or_stop, const Variant& stop, const Variant& step, const DType dtype) {
	try {
		const auto start_ = stop.get_type() == Variant::NIL ? va::VScalar(0) : variant_to_vscalar(start_or_stop);
		const auto stop_ = variant_to_vscalar(stop.get_type() == Variant::NIL ? start_or_stop : stop);
		const auto step_ = variant_to_vscalar(step);

		const auto result = va::arange(
			va::store::default_allocator,
			start_,
			stop_,
			step_,
			dtype
		);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::transpose(const Variant& a, const Variant& permutation) {
	try {
		std::shared_ptr<va::VArray> a_ = variant_as_array(a);

		if (permutation.get_type() == Variant::NIL) {
			return { memnew(NDArray(va::transpose(*a_))) };
		}
		else {
			const auto permutation_ = variant_to_axes(permutation);
			return { memnew(NDArray(va::transpose(*a_, permutation_))) };
		}
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
		[a, b](const std::shared_ptr<va::VArray>& v) {
			return va::swapaxes(*v, a, b);
		}, v
	);
}

Ref<NDArray> nd::moveaxis(const Variant& v, int64_t src, int64_t dst) {
	return map_variants_as_arrays(
		[src, dst](const std::shared_ptr<va::VArray>& v) {
			return va::moveaxis(*v, src, dst);
		}, v
	);
}

Ref<NDArray> nd::flip(const Variant& v, int64_t axis) {
	return map_variants_as_arrays(
		[axis](const std::shared_ptr<va::VArray>& v) {
			return va::flip(*v, axis);
		}, v
	);
}

Ref<NDArray> nd::diagonal(const Variant& v, int64_t offset, int64_t axis1, int64_t axis2) {
	return map_variants_as_arrays(
		[offset, axis1, axis2](const std::shared_ptr<va::VArray>& v) {
			return va::diagonal(*v, offset, axis1, axis2);
		}, v
	);
}

Ref<NDArray> nd::diag(const Variant& v, int64_t offset) {
	try {
		const auto diagonal = variant_as_array(v);

		switch (diagonal->dimension()) {
			case 1: {
				const std::size_t size = diagonal->shape()[0] + std::abs(offset);
				const va::shape_type shape { size, size };

				auto new_array = va::full(
					va::store::default_allocator,
					va::static_cast_scalar(0, diagonal->dtype()),
					shape
				);

				auto new_array_diag = va::diagonal(*new_array, offset, 0, 1);
				va::assign(new_array_diag->data, diagonal->data);

				return { memnew(NDArray(new_array)) };
			}
			case 2:
				return { memnew(NDArray(va::diagonal(*diagonal, offset, 0, 1))) };
			default:
				ERR_FAIL_V_MSG({}, "diag must be called with 1-D or 2-D arrays");
		}
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::trace(const Variant& v, int64_t offset, int64_t axis1, int64_t axis2) {
	return map_variants_as_arrays_with_target([offset, axis1, axis2](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& varray) {
		va::trace(va::store::default_allocator, target, *varray, offset, axis1, axis2);
	}, v);
}

Ref<NDArray> nd::stack(const Variant& v, int64_t axis) {
	return moveaxis(v, 0, axis);
}

Ref<NDArray> nd::unstack(const Variant& v, int64_t axis) {
	return moveaxis(v, axis, 0);
}

Ref<NDArray> concatenate_(nd::DType dtype, const std::vector<std::shared_ptr<va::VArray>>& vector, const std::size_t axis_) {
	ERR_FAIL_COND_V_MSG(!va::is_any_dtype(dtype), {}, "Invalid dtype.");

	auto shape = vector[0]->shape();
	for (auto it = vector.begin() + 1; it != vector.end(); ++it) {
		ERR_FAIL_COND_V_MSG((*it)->dimension() != shape.size(), {}, "Dimensions of given arrays must match.");
		for (int i = 0; i < shape.size(); ++i) {
			if (axis_ == i) continue;
			ERR_FAIL_COND_V_MSG((*it)->shape()[i] != shape[i], {}, "Shapes of given arrays must match.");
		}

		shape[axis_] += (*it)->shape()[axis_];
	}

	if (dtype == nd::DType::DTypeMax) {
		for (auto& array : vector) {
			dtype = va::dtype_common_type_unchecked(dtype, array->dtype());
		}
	}

	auto result = va::empty(va::store::default_allocator, dtype, shape);

	xt::xstrided_slice_vector slice(axis_ + 1);
	std::fill(slice.begin(), slice.end() - 1, xt::all());
	std::size_t current_idx = 0;

	for (auto& array : vector) {
		const auto size_ = array->shape()[axis_];

		slice.back() = xt::xrange(current_idx, current_idx + size_);
		auto write = result->sliced_data(slice);
		va::assign(write, array->data);

		current_idx += size_;
	}

	return { memnew(NDArray(result)) };
}

Ref<NDArray> nd::concatenate(const Variant& v, int64_t axis, DType dtype) {
	try {
		const auto vector = variant_to_vector(v);
		ERR_FAIL_COND_V_MSG(vector.empty(), {}, "Need at least one array to concatenate.");

		if (axis < 0) axis += static_cast<int64_t>(vector[0]->dimension());
		ERR_FAIL_COND_V_MSG(axis < 0 || axis >= vector[0]->dimension(), {}, "Axis out of range.");

		return ::concatenate_(dtype, vector, static_cast<std::size_t>(axis));
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::hstack(const Variant& v, DType dtype) {
	try {
		auto vector = variant_to_vector(v);
		ERR_FAIL_COND_V_MSG(vector.empty(), {}, "Need at least one array to concatenate.");

		for (auto& array : vector) {
			if (array->dimension() == 0) array = array->sliced({ xt::newaxis() });
		}

		return ::concatenate_(dtype, vector, vector[0]->dimension() == 1 ? 0 : 1);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::vstack(const Variant& v, DType dtype) {
	try {
		auto vector = variant_to_vector(v);
		ERR_FAIL_COND_V_MSG(vector.empty(), {}, "Need at least one array to concatenate.");

		for (auto& array : vector) {
			if (array->dimension() == 0) array = array->sliced({ xt::newaxis(), xt::newaxis() });
			else if (array->dimension() == 1) array = array->sliced({ xt::newaxis(), xt::all() });
		}

		return ::concatenate_(dtype, vector, 0);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::tile(const Variant& v, const Variant& reps, bool inner) {
	try {
		const auto array = variant_as_array(v);
		const auto reps_ = variant_to_shape(reps);

		// Let's build the broadcasts.
		// array is sliced as a[..., all, newaxis, all, newaxis, [...]]
		// The result array is reshaped to array shape, with reps_ where array has newaxis.

		const auto result = va::tile(va::store::default_allocator, *array, reps_, inner);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

TypedArray<NDArray> split_(const va::VArray& array, const std::size_t section_size, const size_t axis) {
	ERR_FAIL_COND_V_MSG(array.shape()[axis] % section_size != 0, {}, "Cannot split array equally with this section size.");

	auto godot_array = TypedArray<NDArray>();
	godot_array.resize(static_cast<std::int64_t>(array.shape()[axis]) / section_size);

	xt::xstrided_slice_vector slice(axis + 1);
	std::fill(slice.begin(), slice.end() - 1, xt::all());

	for (std::size_t current_idx = 0; current_idx < godot_array.size(); ++current_idx) {
		slice.back() = xt::range(current_idx * section_size, (current_idx + 1) * section_size);
		godot_array[static_cast<int64_t>(current_idx)] = { memnew(NDArray(array.sliced(slice))) };
	}

	return godot_array;
}

TypedArray<NDArray> split_(const va::VArray& array, const va::strides_type indices, const size_t axis) {
	ERR_FAIL_COND_V_MSG(!std::is_sorted(indices.begin(), indices.end()), {}, "Indices must be sorted.");

	auto godot_array = TypedArray<NDArray>();
	godot_array.resize(indices.size() + 1);

	xt::xstrided_slice_vector slice(axis + 1);
	std::fill(slice.begin(), slice.end() - 1, xt::all());

	for (std::size_t current_idx = 0; current_idx < godot_array.size(); ++current_idx) {
		slice.back() = xt::range(
			current_idx == 0 ? 0 : MAX(0, indices[current_idx - 1]),
			current_idx == godot_array.size() - 1 ? array.shape()[axis] : indices[current_idx]
		);
		godot_array[static_cast<int64_t>(current_idx)] = { memnew(NDArray(array.sliced(slice))) };
	}

	return godot_array;
}

TypedArray<NDArray> split_(const std::shared_ptr<va::VArray>& array, const Variant& indices_or_section_size, int64_t axis) {
	if (axis < 0) axis += static_cast<int64_t>(array->dimension());
	ERR_FAIL_COND_V_MSG(axis < 0 || axis >= array->dimension(), {}, "Axis out of range.");

	if (indices_or_section_size.get_type() == Variant::Type::INT) {
		return ::split_(*array, static_cast<std::size_t>(static_cast<int64_t>(indices_or_section_size)), static_cast<size_t>(axis));
	}

	const auto ints = variant_to_axes(indices_or_section_size);
	return ::split_(*array, ints, static_cast<size_t>(axis));
}

TypedArray<NDArray> nd::split(const Variant& v, const Variant& indices_or_section_size, int64_t axis) {
	try {
		const auto array = variant_as_array(v);
		return split_(array, indices_or_section_size, axis);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

TypedArray<NDArray> nd::hsplit(const Variant& v, const Variant& indices_or_section_size) {
	try {
		const auto array = variant_as_array(v);
		return split_(array, indices_or_section_size, array->dimension() == 1 ? 0 : 1);
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

TypedArray<NDArray> nd::vsplit(const Variant& v, const Variant& indices_or_section_size) {
	return nd::split(v, indices_or_section_size, 0);
}

Ref<NDArray> nd::real(const Variant& v) {
	return map_variants_as_arrays(
		[](const std::shared_ptr<va::VArray>& v) {
			return va::real(v);
		}, v
	);
}

Ref<NDArray> nd::imag(const Variant& v) {
	return map_variants_as_arrays(
		[](const std::shared_ptr<va::VArray>& v) {
			return va::imag(v);
		}, v
	);
}

Ref<NDArray> nd::conjugate(const Variant& a) {
	return VARRAY_MAP1(conjugate, a);
}

Ref<NDArray> nd::angle(const Variant& a) {
	return map_variants_as_arrays_with_target([](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& varray) {
		va::angle(va::store::default_allocator, target, varray);
	}, a);
}

Ref<NDArray> nd::vector_as_complex(const Variant& a, bool keepdims, DType dtype) {
	return map_variants_as_arrays([keepdims, dtype](const std::shared_ptr<va::VArray>& varray) {
		return va::vector_as_complex(va::store::default_allocator, *varray, dtype, keepdims);
	}, a);
}

Ref<NDArray> nd::complex_as_vector(const Variant& a) {
	return map_variants_as_arrays([](const std::shared_ptr<va::VArray>& varray) {
		return va::complex_as_vector(varray);
	}, a);
}

Ref<NDArray> nd::positive(const Variant& a) {
	return VARRAY_MAP1(positive, a);
}

Ref<NDArray> nd::negative(const Variant& a) {
	return VARRAY_MAP1(negative, a);
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

Ref<NDArray> nd::variance(const Variant& a, const Variant& axes) {
	return REDUCTION1(variance, a, axes);
}

Ref<NDArray> nd::standard_deviation(const Variant& a, const Variant& axes) {
	return REDUCTION1(standard_deviation, a, axes);
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

Ref<NDArray> nd::count_nonzero(const Variant& a, const Variant& axes) {
	return reduction([](const va::VArrayTarget& target, const va::axes_type& axes, const va::VArray& array) {
		va::count_nonzero(va::store::default_allocator, target, array.data, axes);
	}, [](const va::VArray& array) { return va::count_nonzero(va::store::default_allocator, array.data); }, axes, a);
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

Ref<NDArray> nd::is_close(const Variant& a, const Variant& b, double_t rtol, double_t atol, bool equal_nan) {
	return map_variants_as_arrays_with_target([rtol, atol, equal_nan](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b) {
		va::is_close(va::store::default_allocator, target, a->data, b->data, rtol, atol, equal_nan);
	}, a, b);
}

Ref<NDArray> nd::is_nan(const Variant& a) {
	return VARRAY_MAP1(is_nan, a);
}

Ref<NDArray> nd::is_inf(const Variant& a) {
	return VARRAY_MAP1(is_inf, a);
}

Ref<NDArray> nd::is_finite(const Variant& a) {
	return VARRAY_MAP1(is_finite, a);
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

Ref<NDArray> nd::bitwise_and(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(bitwise_and, a, b);
}

Ref<NDArray> nd::bitwise_or(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(bitwise_or, a, b);
}

Ref<NDArray> nd::bitwise_xor(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(bitwise_xor, a, b);
}

Ref<NDArray> nd::bitwise_not(const Variant& a) {
	return VARRAY_MAP1(bitwise_not, a);
}

Ref<NDArray> nd::bitwise_left_shift(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(bitwise_left_shift, a, b);
}

Ref<NDArray> nd::bitwise_right_shift(const Variant& a, const Variant& b) {
	return VARRAY_MAP2(bitwise_right_shift, a, b);
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

Ref<NDArray> nd::cross(const Variant& a, const Variant& b, int64_t axisa, int64_t axisb, int64_t axisc) {
	return map_variants_as_arrays_with_target([axisa, axisb, axisc](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b) {
		va::cross(va::store::default_allocator, target, a->data, b->data, axisa, axisb, axisc);
	}, a, b);
}

Ref<NDArray> nd::sliding_window_view(const Variant& array, const Variant& window_shape) {
	try {
		const auto array_ = variant_as_array(array);
		const auto shape_ = variant_to_shape(window_shape);
		const auto result = va::sliding_window_view(*array_, shape_);
		return { memnew(NDArray(result)) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> nd::convolve(const Variant& array, const Variant& kernel) {
	return map_variants_as_arrays_with_target([](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a, const std::shared_ptr<va::VArray>& b) {
		va::convolve(va::store::default_allocator, target, *a, *b);
	}, array, kernel);
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

Ref<NDArray> nd::fft(const Variant& array, const int64_t axis) {
	return map_variants_as_arrays_with_target([axis](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a) {
		va::fft(va::store::default_allocator, target, *a, axis);
	}, array);
}

Ref<NDArray> nd::fft_freq(const int64_t n, const double_t freq) {
	return { memnew(NDArray(va::fft_freq(va::store::default_allocator, n, freq))) };
}

xt::pad_mode pad_mode_to_xt_pad_mode(const nd::PadMode pad_mode) {
	switch (pad_mode) {
		case nd::Constant:
			return xt::pad_mode::constant;
		case nd::Symmetric:
			return xt::pad_mode::symmetric;
		case nd::Reflect:
			return xt::pad_mode::reflect;
		case nd::Wrap:
			return xt::pad_mode::wrap;
		case nd::Edge:
			return xt::pad_mode::edge;
	}

	throw std::runtime_error("Invalid pad mode");
}

Ref<NDArray> nd::pad(const Variant& array, const Variant& pad_width, PadMode pad_mode, const Variant& pad_value) {
	return map_variants_as_arrays_with_target([pad_mode, &pad_value, &pad_width](const va::VArrayTarget& target, const std::shared_ptr<va::VArray>& a) {
		const auto pad_value_scalar = variant_to_vscalar(pad_value);
		const auto pad_width_variant = variant_to_pad_variant(pad_width);
		const auto pad_mode_xt = pad_mode_to_xt_pad_mode(pad_mode);

		std::visit([target, &a, pad_mode_xt, &pad_value_scalar](auto& pad_width) {
			va::pad(va::store::default_allocator, target, *a, pad_width, pad_mode_xt, pad_value_scalar);
		}, pad_width_variant);
	}, array);
}

Ref<NDArray> nd::load(const Variant& variant) {
	try {
		switch (variant.get_type()) {
			case Variant::PACKED_BYTE_ARRAY: {
				const PackedByteArray data = variant;
				const auto result = va::load_npy(reinterpret_cast<const char*>(data.ptr()), data.size());
				return { memnew(NDArray(result)) };
			}
			case Variant::STRING: {
				const String path = variant;
				const PackedByteArray data = FileAccess::get_file_as_bytes(path);
				const auto result = va::load_npy(reinterpret_cast<const char*>(data.ptr()), data.size());
				return { memnew(NDArray(result)) };
			}
			case Variant::OBJECT: {
				if (const auto file_access = Object::cast_to<FileAccess>(variant)) {
					const PackedByteArray data = file_access->get_buffer(file_access->get_length());
					const auto result = va::load_npy(reinterpret_cast<const char*>(data.ptr()), data.size());
					return { memnew(NDArray(result)) };
				}
			}
			default:
				break;
		}

		ERR_FAIL_V_MSG({}, "Unsupported type.");
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

PackedByteArray nd::dumpb(const Variant& array) {
	try {
		const auto array_ = variant_as_array(array);
		auto packed = PackedByteArray();
		packed.resize(static_cast<int64_t>(array_->size()));
		va::util::fill_c_array_flat(packed.ptrw(), array_->data);
		return packed;
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

#undef VARRAY_MAP1
#undef VARRAY_MAP2
#undef VARRAY_MAP3
#undef REDUCTION1
#undef REDUCTION2
