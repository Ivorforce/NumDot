#include "reduce.h"

#include "rearrange.h"
#include <cmath>                                        // for double_t
#include <stdexcept>                                    // for runtime_error
#include <tuple>                                        // for tuple
#include <utility>                                      // for forward, move
#include "vatensor/varray.h"                            // for VArray, axes_...
#include "vcompute.h"                                   // for vreduce, xope...
#include "vpromote.h"                                   // for num_matching_...
#include "xtensor/xiterator.hpp"                        // for operator==
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xmath.hpp"                            // for XTENSOR_REDUC...
#include "xtensor/xnorm.hpp"                            // for norm_l0, norm_l1
#include "xtensor/xoperation.hpp"                       // for logical_and
#include "xtensor/xsort.hpp"                            // for median
#include "xtl/xiterator_base.hpp"                       // for operator!=
namespace xt { namespace evaluation_strategy { struct lazy_type; } }

using namespace va;

// TODO Passing EVS is required because norms don't support it without it, we should make a PR (though it's not bad to explicitly make it lazy).
#define REDUCER_LAMBDA(func) [](auto&& a) { return func(std::forward<decltype(a)>(a), std::tuple<xt::evaluation_strategy::lazy_type>())(); }
#define REDUCER_LAMBDA_NOECS(func) [](auto&& a) { return func(std::forward<decltype(a)>(a)); }
#define REDUCER_LAMBDA_AXES(axes, func) [axes](auto&& a) { return func(std::forward<decltype(a)>(a), axes, std::tuple<xt::evaluation_strategy::lazy_type>()); }
#define REDUCER_LAMBDA_AXES_NOECS(axes, func) [axes](auto&& a) { return func(std::forward<decltype(a)>(a), axes); }

// FIXME These don't support axes yet, see https://github.com/xtensor-stack/xtensor/issues/1555
using namespace xt;
XTENSOR_REDUCER_FUNCTION(va_any, xt::detail::logical_or, bool, true)
XTENSOR_REDUCER_FUNCTION(va_all, xt::detail::logical_and, bool, false)

VScalar va::sum(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_type, VScalar>(
		REDUCER_LAMBDA(xt::sum),
		array.compute_read()
	);
#endif
}

void va::sum(VArrayTarget target, const VArray& array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_common_type>(
		REDUCER_LAMBDA_AXES(axes, xt::sum),
		target, array.compute_read()
	);
#endif
}

VScalar va::prod(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_at_least_int32, VScalar>(
		REDUCER_LAMBDA(xt::prod),
		array.compute_read()
	);
#endif
}

void va::prod(VArrayTarget target, const VArray& array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_common_at_least_int32>(
		REDUCER_LAMBDA_AXES(axes, xt::prod),
		target, array.compute_read()
	);
#endif
}

VScalar va::mean(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VScalar>(
		REDUCER_LAMBDA(xt::mean),
		array.compute_read()
	);
#endif
}

void va::mean(VArrayTarget target, const VArray& array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::mean),
		target, array.compute_read()
	);
#endif
}

VScalar va::median(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_type, VScalar>(
		REDUCER_LAMBDA_NOECS(xt::median),
		array.compute_read()
	);
#endif
}

void va::median(VArrayTarget target, const VArray &array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	throw std::runtime_error("median is not yet supported with a given axis.");

	// TODO xtensor doesn't have median fully implemented yet.
	//  It currently complains with 'unsupported layout' even when just one axis is given.
	//  We have to figure out why even basic tensors are 'dynamic' layout right now.
	//  Then we have to check if it's dynamic layout, and if so, make a row-major copy before calling median.

	// if (axes.size() == 1) {
	// 	// Supported by xtensor.
	// 	auto axis = axes[0];
	// 	va::xoperation_inplace<promote::num_common_type>(
	// 		REDUCER_LAMBDA_AXES_NOECS(axis, xt::median),
	// 		target, array.compute_read()
	// 	);
	// 	return;
	// }

	// Not supported by xtensor. Gotta join the requested axes.
	// const auto joined = join_axes_into_last_dimension(array, axes);
	// constexpr auto axis = -1;
	// va::xoperation_inplace<promote::num_common_type>(
	// 	REDUCER_LAMBDA_AXES_NOECS(axis, xt::median),
	// 	target, joined.compute_read()
	// );
#endif
}

VScalar va::var(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VScalar>(
		REDUCER_LAMBDA(xt::variance),
		array.compute_read()
	);
#endif
}

void va::var(VArrayTarget target, const VArray& array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::variance),
		target, array.compute_read()
	);
#endif
}

VScalar va::std(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VScalar>(
		REDUCER_LAMBDA(xt::stddev),
		array.compute_read()
	);
#endif
}

void va::std(VArrayTarget target, const VArray& array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::stddev),
		target, array.compute_read()
	);
#endif
}

VScalar va::max(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_type, VScalar>(
		REDUCER_LAMBDA(xt::amax),
		array.compute_read()
	);
#endif
}

void va::max(VArrayTarget target, const VArray& array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_common_type>(
		REDUCER_LAMBDA_AXES(axes, xt::amax),
		target, array.compute_read()
	);
#endif
}

VScalar va::min(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_type, VScalar>(
		REDUCER_LAMBDA(xt::amin),
		array.compute_read()
	);
#endif
}

void va::min(VArrayTarget target, const VArray& array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_common_type>(
		REDUCER_LAMBDA_AXES(axes, xt::amin),
		target, array.compute_read()
	);
#endif
}

VScalar va::norm_l0(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VScalar>(
		REDUCER_LAMBDA(xt::norm_l0),
		array.compute_read()
	);
#endif
}

void va::norm_l0(VArrayTarget target, const VArray &array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::norm_l0),
		target, array.compute_read()
	);
#endif
}

VScalar va::norm_l1(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VScalar>(
		REDUCER_LAMBDA(xt::norm_l1),
		array.compute_read()
	);
#endif
}

void va::norm_l1(VArrayTarget target, const VArray &array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::norm_l1),
		target, array.compute_read()
	);
#endif
}

VScalar va::norm_l2(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VScalar>(
		REDUCER_LAMBDA(xt::norm_l2),
		array.compute_read()
	);
#endif
}

void va::norm_l2(VArrayTarget target, const VArray &array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::norm_l2),
		target, array.compute_read()
	);
#endif
}

VScalar va::norm_linf(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VScalar>(
		REDUCER_LAMBDA(xt::norm_linf),
		array.compute_read()
	);
#endif
}

void va::norm_linf(VArrayTarget target, const VArray &array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::norm_linf),
		target, array.compute_read()
	);
#endif
}

bool va::all(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::bool_in_bool_out, bool>(
		REDUCER_LAMBDA_NOECS(xt::all),
		array.compute_read()
	);
#endif
}

void va::all(VArrayTarget target, const VArray &array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::bool_in_bool_out>(
		REDUCER_LAMBDA_AXES(axes, va_all),
		target, array.compute_read()
	);
#endif
}

bool va::any(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::bool_in_bool_out, bool>(
		REDUCER_LAMBDA_NOECS(xt::any),
		array.compute_read()
	);
#endif
}

void va::any(VArrayTarget target, const VArray &array, const axes_type &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::bool_in_bool_out>(
		REDUCER_LAMBDA_AXES(axes, va_any),
		target, array.compute_read()
	);
#endif
}
