#include "reduce.h"

#include <cmath>                                       // for double_t
#include <utility>                                      // for forward
#include "vatensor/varray.h"                            // for VArray, Axes
#include "vcompute.h"
#include "vpromote.h"                                    // for promote
#include "xtensor/xiterator.hpp"                        // for operator==
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xmath.hpp"                            // for amax, amin, mean
#include "xtensor/xnorm.hpp"                            // for norms
#include "xtensor/xtensor_forward.hpp"                            // for xtensor_fixed
#include "xtl/xiterator_base.hpp"                       // for operator!=

using namespace va;

// TODO Passing EVS is required because norms don't support it without it, we should make a PR (though it's not bad to explicitly make it lazy).
#define REDUCER_LAMBDA(func) [](auto&& a) { return func(std::forward<decltype(a)>(a), std::tuple<xt::evaluation_strategy::lazy_type>())(); }
#define REDUCER_LAMBDA_NOECS(func) [](auto&& a) { return func(std::forward<decltype(a)>(a)); }
#define REDUCER_LAMBDA_AXES(axes, func) [axes](auto&& a) { return func(std::forward<decltype(a)>(a), axes, std::tuple<xt::evaluation_strategy::lazy_type>()); }

// FIXME These don't support axes yet, see https://github.com/xtensor-stack/xtensor/issues/1555
using namespace xt;
XTENSOR_REDUCER_FUNCTION(va_any, xt::detail::logical_or, bool, true)
XTENSOR_REDUCER_FUNCTION(va_all, xt::detail::logical_and, bool, false)

VConstant va::sum(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_type, VConstant>(
		REDUCER_LAMBDA(xt::sum),
		array.to_compute_variant()
	);
#endif
}

void va::sum(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_common_type>(
		REDUCER_LAMBDA_AXES(axes, xt::sum),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::prod(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_at_least_int32, VConstant>(
		REDUCER_LAMBDA(xt::prod),
		array.to_compute_variant()
	);
#endif
}

void va::prod(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_common_at_least_int32>(
		REDUCER_LAMBDA_AXES(axes, xt::prod),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::mean(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VConstant>(
		REDUCER_LAMBDA(xt::mean),
		array.to_compute_variant()
	);
#endif
}

void va::mean(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::mean),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::var(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VConstant>(
		REDUCER_LAMBDA(xt::variance),
		array.to_compute_variant()
	);
#endif
}

void va::var(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::variance),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::std(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VConstant>(
		REDUCER_LAMBDA(xt::stddev),
		array.to_compute_variant()
	);
#endif
}

void va::std(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::stddev),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::max(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_type, VConstant>(
		REDUCER_LAMBDA(xt::amax),
		array.to_compute_variant()
	);
#endif
}

void va::max(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_common_type>(
		REDUCER_LAMBDA_AXES(axes, xt::amax),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::min(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_common_type, VConstant>(
		REDUCER_LAMBDA(xt::amin),
		array.to_compute_variant()
	);
#endif
}

void va::min(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_common_type>(
		REDUCER_LAMBDA_AXES(axes, xt::amin),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::norm_l0(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VConstant>(
		REDUCER_LAMBDA(xt::norm_l0),
		array.to_compute_variant()
	);
#endif
}

void va::norm_l0(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::norm_l0),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::norm_l1(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VConstant>(
		REDUCER_LAMBDA(xt::norm_l1),
		array.to_compute_variant()
	);
#endif
}

void va::norm_l1(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::norm_l1),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::norm_l2(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VConstant>(
		REDUCER_LAMBDA(xt::norm_l2),
		array.to_compute_variant()
	);
#endif
}

void va::norm_l2(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::norm_l2),
		target, array.to_compute_variant()
	);
#endif
}

VConstant va::norm_linf(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::num_matching_float_or_default<double_t>, VConstant>(
		REDUCER_LAMBDA(xt::norm_linf),
		array.to_compute_variant()
	);
#endif
}

void va::norm_linf(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_matching_float_or_default<double_t>>(
		REDUCER_LAMBDA_AXES(axes, xt::norm_linf),
		target, array.to_compute_variant()
	);
#endif
}

bool va::all(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::bool_in_bool_out, bool>(
		REDUCER_LAMBDA_NOECS(xt::all),
		array.to_compute_variant()
	);
#endif
}

void va::all(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::bool_in_bool_out>(
		REDUCER_LAMBDA_AXES(axes, va_all),
		target, array.to_compute_variant()
	);
#endif
}

bool va::any(const VArray &array) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	return vreduce<promote::bool_in_bool_out, bool>(
		REDUCER_LAMBDA_NOECS(xt::any),
		array.to_compute_variant()
	);
#endif
}

void va::any(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::bool_in_bool_out>(
		REDUCER_LAMBDA_AXES(axes, va_any),
		target, array.to_compute_variant()
	);
#endif
}
